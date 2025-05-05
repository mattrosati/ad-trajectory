import numpy as np
import torch
from torch.utils.data import Dataset

class GPVAEDataset(Dataset):
    def __init__(self, file_path):
        data = np.load(file_path)  # [N, T, D]
        self.data = torch.tensor(data).float()
        self.T = self.data.shape[1]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {"x": self.data[idx], "t": torch.linspace(0, 1, self.T).unsqueeze(-1).float()}
