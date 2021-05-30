import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
from skimage.io import imread


class img_dataset(Dataset):
    def __init__(self, input_folder, output_folder):
        self.input_paths = glob(input_folder + "*.jpg")
        self.output_paths = glob(output_folder + "*.jpg")
        self.input_paths.sort()
        self.output_paths.sort()

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        data = (imread(self.input_paths[idx]).astype(np.float32) * 2 / 255.) - 1
        label = (imread(self.output_paths[idx]).astype(np.float32) * 2 / 255.) - 1
        data = np.moveaxis(data, -1, 0)
        label = np.moveaxis(label, -1, 0)
        inpt = torch.tensor(data, dtype=torch.float)
        oupt = torch.tensor(label, dtype=torch.float)
        return inpt, oupt
