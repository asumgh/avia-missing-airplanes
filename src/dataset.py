from torch.utils.data import Dataset
import numpy as np
import torch
class Airplane(Dataset):
    def __init__ (self, data, labels, augmentation = None):
        self.X = data
        self.y = labels
        self.aug = augmentation
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        img = self.X[i]
        if self.aug:
            sample = np.array(self.aug(image=img))
        label = self.y[i]
        return torch.tensor(np.moveaxis(sample.reshape(-1)[0]['image'], -1, 0), dtype=torch.float), torch.tensor(label, dtype=torch.float)


class Airplane_test(Dataset):
    def __init__ (self, data, augmentation = None):
        self.X = data
        self.aug = augmentation
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        img = self.X[i]
        if self.aug:
            sample = np.array(self.aug(image=img))
        return torch.tensor(np.moveaxis(sample.reshape(-1)[0]['image'], -1, 0), dtype=torch.float)