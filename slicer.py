#!/usr/bin/env python3
import netCDF4 as nc
import numpy as np
import argparse
import os

def get_file_info(filepath):
    with nc.Dataset(filepath, 'r') as dataset:
        x_shape = dataset.variables['x'].shape
        y_shape = dataset.variables['y'].shape
        file_size = os.path.getsize(filepath) / (1024**3)
        print(f"Arquivo: {filepath}")
        print(f"Shape X: {x_shape}")
        print(f"Shape Y: {y_shape}")
        print(f"Tamanho: {file_size:.2f} GB")
        print(f"Total de amostras: {x_shape[0]:,}")
        return x_shape, y_shape, file_size

def create_memory_efficient_slice(input_file, output_file, max_samples=2000):
    print(f"\n=== Criando slice eficiente ===")
    print(f"Entrada: {input_file}")
    print(f"Saída: {output_file}")
    print(f"Max amostras: {max_samples}")
    x_shape, y_shape, file_size = get_file_info(input_file)
    total_samples = x_shape[0]
    # Seleciona os últimos N samples
    if total_samples <= max_samples:
        print("Arquivo já é pequeno o suficiente!")
        selected_indices = list(range(total_samples))
    else:
        print(f"Selecionando os últimos {max_samples} samples...")
        start_idx = total_samples - max_samples
        selected_indices = list(range(start_idx, total_samples))
    print(f"Selecionadas {len(selected_indices)} amostras")
    with nc.Dataset(input_file, 'r') as src:
        with nc.Dataset(output_file, 'w') as dst:
            # Copia atributos globais
            for attr_name in src.ncattrs():
                dst.setncattr(attr_name, src.getncattr(attr_name))
            new_shape_x = (len(selected_indices),) + x_shape[1:]
            new_shape_y = (len(selected_indices),) + y_shape[1:]
            # Cria dimensões
            dst.createDimension('sample', len(selected_indices))
            dst.createDimension('time', x_shape[1])
            dst.createDimension('lat', x_shape[2])
            dst.createDimension('lon', x_shape[3])
            dst.createDimension('channel', x_shape[4])
            # Cria variáveis principais
            x_var = dst.createVariable('x', src.variables['x'].dtype,
                                       ('sample', 'time', 'lat', 'lon', 'channel'),
                                       fill_value=np.nan, zlib=True, complevel=6)
            y_var = dst.createVariable('y', src.variables['y'].dtype,
                                       ('sample', 'time', 'lat', 'lon', 'channel'),
                                       fill_value=np.nan, zlib=True, complevel=6)
            # Copia atributos das variáveis
            for attr_name in src.variables['x'].ncattrs():
                if attr_name != '_FillValue':
                    x_var.setncattr(attr_name, src.variables['x'].getncattr(attr_name))
            for attr_name in src.variables['y'].ncattrs():
                if attr_name != '_FillValue':
                    y_var.setncattr(attr_name, src.variables['y'].getncattr(attr_name))
            # Copia dados em chunks
            chunk_size = 100
            for i in range(0, len(selected_indices), chunk_size):
                end_i = min(i + chunk_size, len(selected_indices))
                indices_chunk = selected_indices[i:end_i]
                print(f"Copiando amostras {i+1}-{end_i} de {len(selected_indices)}")
                x_chunk = src.variables['x'][indices_chunk]
                y_chunk = src.variables['y'][indices_chunk]
                x_var[i:end_i] = x_chunk
                y_var[i:end_i] = y_chunk
                del x_chunk, y_chunk
            # Cria variáveis de coordenadas
            sample_var = dst.createVariable('sample', 'i8', ('sample',))
            time_var = dst.createVariable('time', 'i8', ('time',))
            lat_var = dst.createVariable('lat', 'i8', ('lat',))
            lon_var = dst.createVariable('lon', 'i8', ('lon',))
            channel_var = dst.createVariable('channel', str, ('channel',))
            # Preenche coordenadas
            sample_var[:] = selected_indices
            time_var[:] = src.variables['time'][:]
            lat_var[:] = src.variables['lat'][:]
            lon_var[:] = src.variables['lon'][:]
            # Copia channels
            channel_data = src.variables['channel'][:]
            for i, ch in enumerate(channel_data):
                channel_var[i] = ch
            # Copia timestamps se existir
            if 'sample_timestamps' in src.variables:
                timestamps_var = dst.createVariable('sample_timestamps', 'i8',
                                                    ('sample', 'time'))
                for i in range(0, len(selected_indices), chunk_size):
                    end_i = min(i + chunk_size, len(selected_indices))
                    indices_chunk = selected_indices[i:end_i]
                    timestamps_var[i:end_i] = src.variables['sample_timestamps'][indices_chunk]
    new_file_size = os.path.getsize(output_file) / (1024**3)
    reduction = file_size / new_file_size
    print(f"\n=== Slice concluído! ===")
    print(f"Arquivo original: {file_size:.2f} GB")
    print(f"Arquivo reduzido: {new_file_size:.2f} GB")
    print(f"Redução: {reduction:.2f}x")
    print(f"Amostras: {total_samples:,} → {len(selected_indices):,}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='output_07_07.nc',
                        help='Arquivo NetCDF de entrada')
    parser.add_argument('--output', '-o', default='output_07_07_sliced.nc',
                        help='Arquivo NetCDF de saída')
    parser.add_argument('--max-samples', '-s', type=int, default=2000,
                        help='Número máximo de amostras')
    args = parser.parse_args()
    if not os.path.exists(args.input):
        print(f"Erro: Arquivo '{args.input}' não encontrado!")
        return
    create_memory_efficient_slice(
        input_file=args.input,
        output_file=args.output,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main()
