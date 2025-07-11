#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import torch
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.preprocessing import MinMaxScaler

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tool.utils import Util
from tool.dataset import NetCDFDataset
from tool.loss import RMSELoss
from tool.train_evaluate import Evaluator
from model.stconvs2s import STConvS2S_C, STConvS2S_R
from torch.utils.data import DataLoader
import torch.nn.functional as F
from captum.attr import IntegratedGradients

data_file = 'data/output_07_07.nc'

# Global variable to control maximum samples per precipitation range
MAX_SAMPLES_PER_RANGE = 10

def sample_by_sample_analysis(model, test_loader, config, device):
    """Prints analysis for a single random sample from the test set, keeping the for loop structure."""
    import random
    print("\n=== Sample-by-Sample Analysis (Random Sample) ===")
    model.eval()
    total_samples = len(test_loader.dataset)
    random_idx = random.randint(0, total_samples - 1)
    batch_size = config.batch
    batch_idx = random_idx // batch_size
    sample_in_batch = random_idx % batch_size

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if i == batch_idx:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                targets_exp = torch.expm1(targets)
                outputs_exp = torch.expm1(outputs)
                # Only analyze the randomly selected sample in this batch
                input_sample = inputs[sample_in_batch]
                target_sample = targets_exp[sample_in_batch]
                output_sample = outputs_exp[sample_in_batch]
                rmse_sample = torch.sqrt(F.mse_loss(outputs[sample_in_batch], targets[sample_in_batch])).item()
                mae_sample = F.l1_loss(outputs[sample_in_batch], targets[sample_in_batch]).item()
                print(f"\nSample {random_idx + 1} (randomly selected):")
                print(f"  Input shape: {input_sample.shape}")
                print(f"  Target shape: {target_sample.shape}")
                print(f"  Output shape: {output_sample.shape}")
                print(f"  RMSE: {rmse_sample:.4f}")
                print(f"  MAE: {mae_sample:.4f}")
                for t in range(target_sample.shape[1]):
                    target_t = target_sample[0, t].cpu().numpy()
                    output_t = output_sample[0, t].cpu().numpy()
                    print(f"    Time step {t+1}:")
                    print(f"      Target - min: {target_t.min():.4f}, max: {target_t.max():.4f}, mean: {target_t.mean():.4f}")
                    print(f"      Output - min: {output_t.min():.4f}, max: {output_t.max():.4f}, mean: {output_t.mean():.4f}")
                break


def get_feature_names_from_dataset(data_file):
    """Extract feature names from NetCDF dataset attributes"""
    try:
        from netCDF4 import Dataset
        with Dataset(data_file, 'r') as ds:
            if hasattr(ds, 'feature_channels'):
                # Extract feature channel mapping from global attributes
                feature_channels = ds.getncattr('feature_channels')
                if isinstance(feature_channels, str):
                    # If stored as string, evaluate it as a dictionary
                    import ast
                    feature_channels = ast.literal_eval(feature_channels)
                
                # Sort by channel number and return feature names
                sorted_channels = sorted(feature_channels.items())
                feature_names = [desc for _, desc in sorted_channels]
                print(f"Found {len(feature_names)} features from dataset metadata:")
                for i, name in enumerate(feature_names):
                    print(f"  Channel {i}: {name}")
                return feature_names
            else:
                print("No feature_channels attribute found in dataset")
                return None
    except Exception as e:
        print(f"Error reading feature names from dataset: {e}")
        return None


def analyze_feature_importance_by_range(model, test_input_tensor, test_y_tensor, features_names, device, examples_dir, range_name, range_mask):
    """Analyze feature importance for a specific precipitation range"""
    if range_mask.sum() == 0:
        print(f"No samples found in range {range_name}, skipping...")
        return None
        
    # Select samples in this range
    range_inputs = test_input_tensor[range_mask]
    
    # Limit the number of samples per range using the global variable
    if range_inputs.shape[0] > MAX_SAMPLES_PER_RANGE:
        indices = torch.randperm(range_inputs.shape[0])[:MAX_SAMPLES_PER_RANGE]
        range_inputs = range_inputs[indices]
        print(f"Analyzing {range_inputs.shape[0]} samples in range {range_name} (limited from {range_mask.sum().item()})")
    else:
        print(f"Analyzing {range_inputs.shape[0]} samples in range {range_name}")
    
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            output = self.model(x)
            # Return mean of the first channel across all spatial and temporal dimensions
            return output[:, 0].mean(dim=(1, 2, 3))
    
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()
    
    ig = IntegratedGradients(wrapped_model)
    
    try:
        baseline = torch.zeros_like(range_inputs)
        attr = ig.attribute(range_inputs, baseline)
        attr = attr.detach().cpu().numpy()
        
        # Channel importance
        importances = np.mean(attr, axis=(0, 2, 3, 4))
        print(f"\nChannel Importances for {range_name}:")
        for i, name in enumerate(features_names[:len(importances)]):
            print(f"{name}: {importances[i]:.6f}")
        
        # Channel importance visualization
        if len(importances) <= len(features_names):
            plt.figure(figsize=(12, 6))
            x_pos = np.arange(len(importances))
            plt.bar(x_pos, importances)
            plt.xticks(x_pos, features_names[:len(importances)], rotation=45, ha='right')
            plt.xlabel("Features")
            plt.ylabel("Attribution")
            plt.title(f"Feature Importance - {range_name}")
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(examples_dir, f"feature_importance_{range_name.replace('-', '_').replace('+', 'plus').replace(' ', '_')}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Feature importance plot saved to: {plot_path}")
            
        # Spatial importance
        spatial_importances = np.mean(attr, axis=(0, 1, 2))
        print(f"Spatial importance - Max: {spatial_importances.max():.6f}, Min: {spatial_importances.min():.6f}")
        
        # Spatial importance visualization
        plt.figure(figsize=(8, 6))
        plt.imshow(spatial_importances, cmap="viridis")
        plt.colorbar(label="Importance")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Spatial Importances - {range_name}")
        
        # Save spatial plot
        spatial_plot_path = os.path.join(examples_dir, f"spatial_importance_{range_name.replace('-', '_').replace('+', 'plus').replace(' ', '_')}.png")
        plt.savefig(spatial_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Spatial importance plot saved to: {spatial_plot_path}")
        
        # Temporal importance
        temporal_importances = np.mean(attr, axis=(0, 1, 3, 4))
        print(f"Temporal Importances for {range_name}:")
        for i, importance in enumerate(temporal_importances):
            print(f"Timestep {i}: {importance:.6f}")
            
        # Temporal importance visualization
        timestep_names = [f"Timestep {i+1}" for i in range(len(temporal_importances))]
        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(timestep_names))
        plt.bar(x_pos, temporal_importances, align="center")
        plt.xticks(x_pos, timestep_names, rotation=45)
        plt.xlabel("Timesteps")
        plt.ylabel("Attribution")
        plt.title(f"Temporal Importances - {range_name}")
        plt.tight_layout()
        
        # Save temporal plot
        temporal_plot_path = os.path.join(examples_dir, f"temporal_importance_{range_name.replace('-', '_').replace('+', 'plus').replace(' ', '_')}.png")
        plt.savefig(temporal_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Temporal importance plot saved to: {temporal_plot_path}")
            
        return attr
        
    except Exception as e:
        print(f"Error in feature importance analysis for {range_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_feature_importance(model, test_dataset, device, examples_dir, num_samples=50):
    """Analyze feature importance using Captum IntegratedGradients by precipitation ranges"""
    print("\n=== Feature Importance Analysis by Precipitation Ranges ===")
    
    # Try to get feature names from the dataset
    features_names = get_feature_names_from_dataset(data_file)
    if not features_names:
        # Fallback to default names if dataset doesn't contain feature info
        features_names = [
            "densidade_flashes", "glaciacao_topo_nuvem",
            "li_proxy", "movimento_vertical", "profundidade_nuvens",
            "tamanho_particulas", "temperatura_topo_nuvem", "textura_local_profundidade"
        ]
    
    # Use more samples for range analysis
    test_input_tensor = test_dataset.X[:num_samples].to(device)
    test_y_tensor = test_dataset.y[:num_samples]
    
    print(f"Analyzing {num_samples} samples with shape: {test_input_tensor.shape}")
    
    # Convert y values back to original scale for range classification
    # Since y is in log1p scale, convert back
    y_original = torch.expm1(test_y_tensor)
    
    # Get max precipitation value across all dimensions for each sample
    y_max_per_sample = y_original.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0].max(dim=1)[0]  # shape: (num_samples,)
    
    # Define precipitation ranges (in mm)
    ranges = {
        "0-5mm": (y_max_per_sample < 5),
        "5-25mm": ((y_max_per_sample >= 5) & (y_max_per_sample < 25)),
        "25-50mm": ((y_max_per_sample >= 25) & (y_max_per_sample < 50)),
        "50+mm": (y_max_per_sample >= 50)
    }
    
    print(f"\nSample distribution by precipitation ranges:")
    for range_name, mask in ranges.items():
        count = mask.sum().item()
        print(f"  {range_name}: {count} samples ({count/num_samples*100:.1f}%)")
    
    # Run analysis for each range
    results = {}
    for range_name, range_mask in ranges.items():
        print(f"\n{'='*50}")
        print(f"Analyzing range: {range_name}")
        print(f"{'='*50}")
        
        result = analyze_feature_importance_by_range(
            model, test_input_tensor, test_y_tensor, features_names, 
            device, examples_dir, range_name, range_mask
        )
        if result is not None:
            results[range_name] = result
    
    return results


def run_model_on_sample():
    """Run the trained model on a few samples and visualize predictions vs ground truth"""
    # Configuration
    class Config:
        def __init__(self):
            self.model = 'stconvs2s-r'
            self.pre_trained = '/home/ge28/Desktop/Python/trab_final/APRENDIZADO_T2/output/full-dataset/checkpoints/stconvs2s-r/cfsr_step5_4_20250709-164829.pth.tar'
            self.step = 5
            self.num_layers = 3
            self.hidden_dim = 32
            self.kernel_size = 5
            self.batch = 5
            self.workers = 4
            self.cuda = 0
            self.small_dataset = False
            self.chirps = False
            self.verbose = True
            self.no_seed = False

    config = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using {'GPU: ' + torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print("=== Running Model on Sample Data ===")
    print(f"Model: {config.model.upper()}")
    print(f"Pre-trained model: {config.pre_trained}")
    print(f"Step ahead prediction: {config.step}")

    if not os.path.exists(config.pre_trained):
        print(f"Error: Pre-trained model not found: {config.pre_trained}")
        return

    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        return

    print(f"Loading data from: {data_file}")

    try:
        # Set seed for reproducibility
        if not config.no_seed:
            seed = 1000
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True

        # Load dataset
        ds = xr.open_mfdataset(data_file).load()
        if config.small_dataset:
            ds = ds[dict(sample=slice(0, 500))]

        # Use fractions for splits (fixes empty train set bug and type error)
        validation_split = 0.2
        test_split = 0.2
        train_dataset = NetCDFDataset(ds, test_split=test_split, validation_split=validation_split)
        test_dataset = NetCDFDataset(ds, test_split=test_split, validation_split=validation_split, is_test=True)

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")

        # Apply log1p transform for precipitation (Y)
        for dataset_name, dataset in zip(["train", "test"], [train_dataset, test_dataset]):
            y_np = dataset.y.detach().cpu().numpy()
            y_log = np.log1p(y_np)
            dataset.y = torch.tensor(y_log, dtype=dataset.y.dtype, device=dataset.y.device)
            if config.verbose:
                print(f"Applied log1p to Y in {dataset_name} set. Before: min={y_np.min():.4f}, max={y_np.max():.4f} | After: min={y_log.min():.4f}, max={y_log.max():.4f}")

        # Apply MinMaxScaler feature scaling
        for channel_idx in range(train_dataset.X.shape[1]):
            # Fit scaler on training data
            train_data = train_dataset.X[:, channel_idx].detach().cpu().numpy()
            original_shape = train_data.shape
            reshaped = train_data.reshape(-1, 1)
            scaler = MinMaxScaler().fit(reshaped)
            scaled_train = scaler.transform(reshaped).reshape(original_shape)
            train_dataset.X[:, channel_idx] = torch.tensor(scaled_train, dtype=train_dataset.X.dtype, device=train_dataset.X.device)

            # Apply to test dataset
            test_data = test_dataset.X[:, channel_idx].detach().cpu().numpy()
            reshaped = test_data.reshape(-1, 1)
            scaled_test = scaler.transform(reshaped).reshape(test_data.shape)
            test_dataset.X[:, channel_idx] = torch.tensor(scaled_test, dtype=test_dataset.X.dtype, device=test_dataset.X.device)

        print(f"Data shapes - X: {test_dataset.X.shape}, Y: {test_dataset.y.shape}")

        # Create data loader for test data (small batch to see individual samples)
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=config.batch, num_workers=config.workers)

        # Create model
        models = {
            'stconvs2s-r': STConvS2S_R,
            'stconvs2s-c': STConvS2S_C,
        }
        if config.model not in models:
            print(f"Error: Unknown model {config.model}")
            return

        model_builder = models[config.model]
        model = model_builder(train_dataset.X.shape, config.num_layers, config.hidden_dim, config.kernel_size, device, 0.0, config.step)
        model.to(device)

        # Create loss function and optimizer
        criterion = RMSELoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-6)

        # Create utility for visualization
        util = Util(config.model, 'sample-test', version=0)

        # Create evaluator and load checkpoint
        evaluator = Evaluator(model, criterion, optimizer, test_loader, device, util, config.step)

        print(f"\nLoading pre-trained model from: {config.pre_trained}")
        best_epoch, val_loss = evaluator.load_checkpoint(config.pre_trained)
        if best_epoch == 0:
            print("Failed to load the model checkpoint!")
            return

        print(f"Successfully loaded model from epoch {best_epoch} with validation RMSE: {val_loss:.4f}")
        print("\nRunning model evaluation and creating visualizations...")
        test_rmse, test_mae = evaluator.eval(is_test=True, is_chirps=config.chirps)

        print(f"\n=== Results ===")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test MAE: {test_mae:.4f}")

        sample_by_sample_analysis(model, test_loader, config, device)
        
        # Get examples directory before calling analyze_feature_importance
        examples_dir = util.get_examples_dir()
        analyze_feature_importance(model, test_dataset, device, examples_dir, num_samples=100)

        print(f"\n=== Visualization Files Created ===")
        # examples_dir is already available from above
        if os.path.exists(examples_dir):
            files = os.listdir(examples_dir)
            print(f"Check the following directory for visualization files:")
            print(f"  {examples_dir}")
            print(f"Files created: {files}")
        else:
            print("No visualization files were created.")

        print("\n=== Analysis Complete ===")
        print("The visualization files show:")
        print("  - Input: The input sequence used for prediction")
        print("  - Ground Truth: The actual values")
        print("  - Prediction: The model's predictions")
        print("  - Feature Importance by Range: Importance of each input channel/feature for different precipitation levels")
        print("    * 0-5mm (light precipitation)")
        print("    * 5-25mm (moderate precipitation)")
        print("    * 25-50mm (heavy precipitation)")
        print("    * 50+mm (very heavy precipitation)")
        print("  - Spatial Importance by Range: Geographic importance maps (lat/lon) for each precipitation level")
        print("  - Temporal Importance by Range: Importance across time steps for each precipitation level")
        print("Compare these to see how feature importance changes across precipitation intensities!")

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_model_on_sample()
