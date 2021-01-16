import io
import os
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    """Chest Xray dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.name_frame = pd.read_csv(csv_file, sep=",", usecols=range(1))
        self.label_frame = pd.read_csv(csv_file, sep=",", usecols=range(1, 2))
        self.root_dir = root_dir

    def __len__(self):
        return len(self.name_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.name_frame.iloc[idx, 0])
        image = io.imread(img_name)
        labels = self.label_frame.iloc[idx, 0]
        sample = {'image': image, 'label': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample
