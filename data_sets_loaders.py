from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import os
from os import listdir
import numpy as np
import PIL
from PIL import Image

class CombinedSRDataset(Dataset):
  def __init__(self, lr_image_names, hr_image_names, isTrain=True, cropped_dir="./cropped_images"):
    self.lr_image_names = lr_image_names
    self.hr_image_names = hr_image_names
    self.cropped_dir = cropped_dir
    self.datasetType = "train" if isTrain else "test"
        
  def __getitem__(self, index):
    lr_img = Image.open("{}/{}/LR/".format(self.cropped_dir, self.datasetType) + self.lr_image_names[index])
    hr_img = Image.open("{}/{}/HR/".format(self.cropped_dir, self.datasetType) + self.hr_image_names[index])
    lr_img_arr = np.array(lr_img, dtype=np.float32) / 255.0
    hr_img_arr = np.array(hr_img, dtype=np.float32) / 255.0
    return (
            np.transpose(lr_img_arr, (2, 0, 1)),
            np.transpose(hr_img_arr, (2, 0, 1))
        )

    
  def __len__(self):
    return len(self.lr_image_names)

def get_train_val_test_dataloaders(batch_size, cropped_dir="./cropped_images"):
		train_set = CombinedSRDataset(sorted(os.listdir("{}/{}/LR".format(cropped_dir, "train"))), sorted(os.listdir("{}/{}/HR".format(cropped_dir, "train"))))
		test_set = CombinedSRDataset(sorted(os.listdir("{}/{}/LR".format(cropped_dir, "test"))), sorted(os.listdir("{}/{}/HR".format(cropped_dir, "test"))), isTrain=False)

		# Create the training and validation indices to split the train_set
		dataset_size = len(train_set)
		indices = list(range(dataset_size))
		np.random.shuffle(indices)
		split = int(len(indices) * 0.8) #split at 80%

		train_indices, val_indices = indices[:split], indices[split:]
		train_sampler = SubsetRandomSampler(train_indices)
		val_sampler = SubsetRandomSampler(val_indices)

		# Create a Dataloader for each train, val, and test
		train_dl = DataLoader(train_set, batch_size=batch_size, 
													num_workers=1, sampler=train_sampler)
		val_dl = DataLoader(train_set, batch_size=batch_size,
													num_workers=1, sampler=val_sampler)
		test_dl = DataLoader(test_set, batch_size=batch_size,
													num_workers=1)
		return train_dl, val_dl, test_dl