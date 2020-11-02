import torch
import os
from skimage import io, transform
import pandas as pd
import numpy as np 
from torch.utils.data import Dataset

class AptosDataset(Dataset):
    "Change the Constructor as per your Dataset"
    def  __init__(self, csv_filepath, root_dir, transform = None ):

        self.labels = pd.read_csv(csv_filepath)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
        img_name = img_name + ".png"

        image = io.imread(img_name)

        y = self.labels.iloc[idx, 1]

        if self.transform:
            #print("transform the image for augmentation")
            img, label = self.transform((image, y))



        return img, label





