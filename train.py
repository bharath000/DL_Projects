from dataloaders.aptos import AptosDataset
from CustomTransforms.customtransform import Rescale, RandomShear, RandomShift, RandomRotate, RandomScaling, Sharpen, Brightness, Contrast
from torchvision import transforms, utils
from utility.grid_plot import plot_images 
from scipy import ndimage
import numpy as np 
from torch.utils.data import DataLoader
import torch

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

##### Spliting Trainning Data into Train and Validation data #####

##### LOADING TRAIN DATA ######

aptos_dataset = AptosDataset('D:/MSCS/ADL/Data/train.csv', 'D:/MSCS/ADL/Data/train_images/',
                                transform=transforms.Compose([
                                               Brightness(20),
                                               
                                               #Contrast(0.2),
                                               Rescale((512,512)),
                                               Sharpen(0.5),
                                               #Sharpen(2), 
                                               RandomShear(0.2),
                                               #RandomShift((10, -20)),
                                               #RandomRotate(180.0),
                                               #RandomScaling((1.2, 1.2)),
                                               #RandomCrop(224),
                                               #ToTensor()
                                               #ToNormalize()
                                           ]))

train_generator = DataLoader(aptos_dataset, batch_size=4,
                        shuffle=True, num_workers=0)




##### LOADING Validation Data #####


#sample, y  = aptos_dataset[0]





for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch,len(sample_batched))
    if i_batch == 4:
        
        plot_images(sample_batched)
        
        break


#plot_images(sample)
#print(sample.shape, y)


