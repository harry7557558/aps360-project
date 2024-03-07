class MyDataset(Dataset):
  def __init__(self, lr_paths_dir, hr_paths_dir):
    self.lr_paths_dir = lr_paths_dir
    self.hr_paths_dir = hr_paths_dir
    self.lr_images_names = sorted([s for s in lr_paths_dir])
    self.hr_images_names = sorted([s for s in hr_paths_dir])
        
  def __getitem__(self, index):
    lr_img = Image.open( self.lr_paths_dir + "/" + self.lr_images_names[index])
    hr_img = Image.open( self.hr_paths_dir + "/" + self.hr_images_names[index])
    return lr_img, hr_img
    
  def __len__(self):
    return len(self.lr_paths_dir)

def get_train_val_test_dataloaders(batch_size):
		train_set = MyDataset(
    sorted(os.listdir("{}/{}/LR".format(CROPPED_DATASET_IMAGES_DIR, "train"))), sorted(os.listdir("{}/{}/HR".format(CROPPED_DATASET_IMAGES_DIR, "train"))))
		test_set = MyDataset(
		sorted(os.listdir("{}/{}/LR".format(CROPPED_DATASET_IMAGES_DIR, "test"))), sorted(os.listdir("{}/{}/HR".format(CROPPED_DATASET_IMAGES_DIR, "test"))))

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