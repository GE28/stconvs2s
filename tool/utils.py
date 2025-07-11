import numpy as np
import pandas as pd
import smtplib
import os
import time as tm
from datetime import datetime
from configparser import ConfigParser
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') #non-interactive backends for png files
from matplotlib.ticker import MaxNLocator
import torch

class Util:
    def __init__(self, model_descr, dataset_type='notebooks', version=0, prefix=''):
        current_time = datetime.now()
        self.model_descr = model_descr
        self.start_time = current_time.strftime('%d/%m/%Y %H:%M:%S')
        self.start_time_timestamp = tm.time()
        self.version = str(version)
        prefix = prefix.lower() + '_' if prefix.strip() else ''
        self.base_filename =  prefix + self.version + '_' + current_time.strftime('%Y%m%d-%H%M%S')
        self.project_dir = str(Path(__file__).absolute().parent.parent)
        self.output_dir = os.path.join(self.project_dir, 'output', dataset_type)
        
    def plot(self, data, columns_name, x_label, y_label, title, enable=True, inline=False):
        if (enable):
            df = pd.DataFrame(data).T    
            df.columns = columns_name
            df.index += 1
            plot = df.plot(linewidth=2, figsize=(15,8), color=['darkgreen', 'orange'], grid=True);
            train = columns_name[0]
            val = columns_name[1]
            # find position of lowest validation loss
            idx_min_loss = df[val].idxmin()
            plot.axvline(idx_min_loss, linestyle='--', color='r',label='Best epoch');
            plot.legend();
            plot.set_xlim(0, len(df.index)+1);
            plot.xaxis.set_major_locator(MaxNLocator(integer=True))
            plot.set_xlabel(x_label, fontsize=12);
            plot.set_ylabel(y_label, fontsize=12);
            plot.set_title(title, fontsize=16);
            if (not inline):
                plot_dir = self.__create_dir('plots')
                filename = os.path.join(plot_dir, self.base_filename + '.png')
                plot.figure.savefig(filename, bbox_inches='tight');
        
    def send_email(self, model_info, enable=True):
        if (enable):
            config = ConfigParser()
            config.read(os.path.join(self.project_dir, 'config/mail_config.ini'))
            server = config.get('mailer','server')
            port = config.get('mailer','port')
            login = config.get('mailer','login')
            password = config.get('mailer', 'password')
            to = config.get('mailer', 'receiver')

            subject = 'Experiment execution [' + self.model_descr + ']'
            text = 'This is an email message to inform you that the python script has completed.'
            message = text + '\n' + str(self.get_time_info()) + '\n' + str(model_info)

            smtp = smtplib.SMTP_SSL(server, port)
            smtp.login(login, password)

            body = '\r\n'.join(['To: %s' % to,
                                'From: %s' % login,
                                'Subject: %s' % subject,
                                '', message])
            try:
                smtp.sendmail(login, [to], body)
                print ('email sent')
            except Exception:
                print ('error sending the email')

            smtp.quit()
    
    def save_loss(self, train_losses, val_losses, enable=True):
        if (enable):
            losses_dir = self.__create_dir('losses')
            train_dir, val_dir = self.__create_train_val_dir_in(losses_dir)
            train_filename = os.path.join(train_dir, self.base_filename + '.txt')
            val_filename = os.path.join(val_dir, self.base_filename + '.txt')
            np.savetxt(train_filename, train_losses, delimiter=",", fmt='%g')
            np.savetxt(val_filename, val_losses, delimiter=",", fmt='%g')
            
    def save_examples(self, inputs, target, output, step):
        # Visualize all channels and all time steps for each tensor
        # Assumes input shape: [batch, channels, time, height, width]
        def plot_tensor_grid(tensor, name, step, examples_dir):
            batch_idx = 0  # Only plot the first sample in the batch for clarity
            channels = tensor.shape[1]
            time_steps = tensor.shape[2]
            # Compute global min and max for consistent color range
            tensor_np = tensor[batch_idx].cpu().numpy()  # shape: [channels, time, H, W]
            vmin = tensor_np.min()
            vmax = tensor_np.max()
            # Handle 1D cases for channels or time_steps
            if channels == 1 and time_steps == 1:
                fig, axes = plt.subplots(1, 1, figsize=(4, 4))
                axes_grid = np.array([[axes]])
            elif channels == 1:
                fig, axes = plt.subplots(1, time_steps, figsize=(3*time_steps, 3))
                axes_grid = np.array(axes).reshape((1, time_steps))
            elif time_steps == 1:
                fig, axes = plt.subplots(channels, 1, figsize=(3, 3*channels))
                axes_grid = np.array(axes).reshape((channels, 1))
            else:
                fig, axes = plt.subplots(channels, time_steps, figsize=(3*time_steps, 3*channels))
                axes_grid = axes
            cmap = 'YlGnBu' if self.base_filename.startswith('chirps') else 'viridis'
            for c in range(channels):
                for t in range(time_steps):
                    ax = axes_grid[c, t]
                    img = tensor[batch_idx, c, t].cpu().numpy()
                    ax.imshow(np.flipud(img), cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set_title(f"Ch {c}, T {t}")
                    ax.axis('off')
            fig.suptitle(f"{name}")
            filename = os.path.join(examples_dir, f"{name}_grid_{self.base_filename}.png")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fig.savefig(filename, dpi=300)
            plt.close(fig)

        examples_dir = self.__create_dir('examples', include_model_descr=False)
        plot_tensor_grid(inputs, 'input', step, examples_dir)
        plot_tensor_grid(target, 'ground_truth', step, examples_dir)
        plot_tensor_grid(output, 'prediction', step, examples_dir)
    
    def get_checkpoint_filename(self):
        check_dir = self.__create_dir('checkpoints')
        filename = os.path.join(check_dir, self.base_filename + '.pth.tar')
        return filename
        
    def to_readable_time(self, timestamp):
        hours = int(timestamp / (60 * 60))
        minutes = int((timestamp % (60 * 60)) / 60)
        seconds = timestamp % 60.
        return f'{hours}:{minutes:>02}:{seconds:>05.2f}'
        
    def get_time_info(self):
        end_time = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        end_time_timestamp = tm.time()
        elapsed_time = end_time_timestamp - self.start_time_timestamp
        elapsed_time = self.to_readable_time(elapsed_time)
        time_info = {'model': self.model_descr,
                      'version': self.version,
                      'start_time': self.start_time,
                      'end_time': end_time,
                      'elapsed_time': elapsed_time}
        return time_info
       
    def get_mask_land(self):
        """
        Original chirps dataset has no ocean data, 
        so this mask is required to ensure that only land data is considered
        """
        filename = os.path.join(self.project_dir, 'data', 'chirps_mask_land.npy')
        mask_land = np.load(filename)
        mask_land = torch.from_numpy(mask_land).float()
        return mask_land
        
    @staticmethod         
    def generate_list_from(integer, size=3):
        if isinstance(integer,int):
            return [integer] * size
        return integer        
        
    def __create_train_val_dir_in(self, dir_path):
        train_dir = os.path.join(dir_path, 'train')
        os.makedirs(train_dir, exist_ok=True)
        val_dir = os.path.join(dir_path, 'val')   
        os.makedirs(val_dir, exist_ok=True)     
        return train_dir, val_dir
    
    def __create_dir(self, dir_name, include_model_descr=True):
        if include_model_descr:
            new_dir = os.path.join(self.output_dir, dir_name, self.model_descr)
        else:
            new_dir = os.path.join(self.output_dir, dir_name)
        os.makedirs(new_dir, exist_ok=True)
        return new_dir
        
    def __create_image_plot(self, tensor, ax, i, j, index, step, ax_input=False):
        cmap = 'YlGnBu' if self.base_filename.startswith('chirps') else 'viridis'
        tensor_numpy = tensor[0,:,index,:,:].squeeze().cpu().numpy()
        if step == 5 or ax_input:
            # Select the first time/step slice if tensor_numpy has 3 dimensions
            if tensor_numpy.ndim == 3:
                tensor_numpy = tensor_numpy[0]
            ax[j].imshow(np.flipud(tensor_numpy), cmap=cmap)
            ax[j].get_xaxis().set_visible(False)
            ax[j].get_yaxis().set_visible(False)
        else:
            ax[i][j].imshow(np.flipud(tensor_numpy), cmap=cmap)
            ax[i][j].get_xaxis().set_visible(False)
            ax[i][j].get_yaxis().set_visible(False)        
        return ax
        
    def __save_image_plot(self, figure, folder, name, step, fig_input=False):
        y = 0.7 if (step == 5 or fig_input) else 0.9
        figure.suptitle(name, y=y)
        filename = os.path.join(folder, name + '_' + self.base_filename + '.png') 
        figure.savefig(filename, dpi=300)
    
    def get_examples_dir(self):
        """Return the directory where example images are saved (public method)."""
        return self.__create_dir('examples', include_model_descr=False)