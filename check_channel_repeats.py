#!/usr/bin/env python3
import netCDF4 as nc
import numpy as np
import torch
import random
import os

# Path to NetCDF file (update if needed)
data_file = '/home/ge28/Desktop/Python/trab_final/APRENDIZADO_T2/data/output_07_07_orig.nc'
if not os.path.exists(data_file):
    data_file = 'output_07_07.nc'
    if not os.path.exists(data_file):
        raise FileNotFoundError('NetCDF file not found!')

# Open NetCDF file
ds = nc.Dataset(data_file, 'r')

gt_var_names = [k for k in ds.variables.keys() if 'ground_truth' in k or 'target' in k or 'y' == k]

if not gt_var_names:
    print('Could not automatically find ground truth variables. Please update variable names in the script.')
    print('Variables in file:', list(ds.variables.keys()))
    ds.close()
    exit(1)

gt_var = ds.variables[gt_var_names[0]]

gt = np.array(gt_var)

print(f"Ground truth shape: {gt.shape}")

N = 50
num_samples = gt.shape[0]
sample_indices = random.sample(range(num_samples), min(N, num_samples))

for idx in sample_indices:
    y_sample = gt[idx]
    print(f"\nSample {idx}:")
    y_channels = y_sample.reshape(y_sample.shape[0], -1)
    y_repeats = False
    for i in range(y_channels.shape[0]):
        for j in range(i+1, y_channels.shape[0]):
            if np.allclose(y_channels[i], y_channels[j]):
                print(f"  y channels {i} and {j} are identical.")
                y_repeats = True
    if not y_repeats:
        print("  No repeated channels in y.")

ds.close()
print("\nCheck complete.")
