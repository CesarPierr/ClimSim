import os
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import logging
from typing import Dict, Any

from climsim_utils.data_utils import data_utils

logging.basicConfig(level=logging.INFO)

class ClimSimDataset(Dataset):
    def __init__(self, 
                    base_dir: str,
                    dataset_dir: str,
                    grid_file: str,
                    normalize: bool = True,
                    data_split: str = 'train',
                    regexps: list = None,
                    cnn_reshape: bool = False,
                    stride_sample: int = 1,):

        super().__init__()
        self.base_dir = Path(base_dir).resolve()
        dataset_dir = Path(dataset_dir).resolve()
        norm_path = dataset_dir / 'ClimSim' / 'preprocessing' / 'normalizations'
        self.normalize = normalize
        self.cnn_reshape = cnn_reshape

        try:
            grid_path = self.base_dir / grid_file
            self.grid_ds = xr.open_dataset(grid_path, engine='netcdf4')

            input_mean = xr.open_dataset(norm_path / 'inputs' / 'input_mean.nc', engine='netcdf4')
            input_max  = xr.open_dataset(norm_path / 'inputs' / 'input_max.nc',  engine='netcdf4')
            input_min  = xr.open_dataset(norm_path / 'inputs' / 'input_min.nc',  engine='netcdf4')
            output_scale = xr.open_dataset(norm_path / 'outputs' / 'output_scale.nc', engine='netcdf4')
        except Exception as e:
            logging.error(f"Error loading required files: {e}")
            raise

        self.data = data_utils(
            grid_info    = self.grid_ds,
            input_mean   = input_mean,
            input_max    = input_max,
            input_min    = input_min,
            output_scale = output_scale,
            ml_backend   = "pytorch",
            normalize    = normalize,
        )
        self.data.set_to_v1_vars()
        
        self.data.data_path = str(self.base_dir / 'train')+'/'
        self.data.set_regexps(
            data_split=data_split,
            regexps=regexps,
        )
        self.data.set_stride_sample(data_split, stride_sample=stride_sample)
        self.data.set_filelist(data_split)
        self.file_list = self.data.get_filelist(data_split)
        if not self.file_list:
            raise ValueError(f"No files found for split='{data_split}' under {self.data.data_path}")
        logging.info(f"[{data_split} split] Found {len(self.file_list)} netCDF files.")

    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:

        file_path = self.file_list[idx]
        ds_input  = self.data.get_input(file_path)
        ds_target = self.data.get_target(file_path)


        if self.normalize:
            ds_input  = (ds_input - self.data.input_mean) / (self.data.input_max - self.data.input_min)
            ds_target = ds_target * self.data.output_scale

        x = ds_input.stack(batch=('ncol',)).to_stacked_array('mlvar', sample_dims=['batch']).values
        y = ds_target.stack(batch=('ncol',)).to_stacked_array('mlvar', sample_dims=['batch']).values

        if self.cnn_reshape:
            x_cnn = self.data.reshape_input_for_cnn(x)  # shape (N, 60, 6)
            y_cnn = self.data.reshape_target_for_cnn(y) # shape (N, 60, 10)

            x = torch.from_numpy(x_cnn).float()
            y = torch.from_numpy(y_cnn).float()  
        else:
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()

        return {
            "input": x,
            "target": y,  
            "file_path": str(file_path),
        }

if __name__ == "__main__":

    base_dir = "/home/deemel/pierre/dataset"  

    dataset = ClimSimDataset(
        base_dir = base_dir,
        grid_file = "ClimSim_low-res_grid-info.nc",
        normalize = True,
        data_split = "train",  
        regexps = [
            "E3SM-MMF.mli.000[1234567]-*-*-*.nc", 
            "E3SM-MMF.mli.0008-01-*-*.nc",
        ],
        cnn_reshape = True,
    )
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)

    batch = next(iter(dataloader))
    x, y = batch["input"], batch["target"]
    print(f"Input shape: {x.shape}, Target shape: {y.shape}")
    