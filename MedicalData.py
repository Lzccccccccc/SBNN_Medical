import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MedicalDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'data': self.data[idx],
            'label': self.labels[idx]
        }

        if self.transform:
            sample["data"] = self.transform(sample["data"])
        return sample
