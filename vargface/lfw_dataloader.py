
import os
import random
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


# Set the seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LFWDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self._prepare_data()

    def _prepare_data(self):
        for idx, class_dir in enumerate(os.listdir(self.root_dir)):
            class_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_path):
                self.class_to_idx[class_dir] = idx
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_train_val_loaders(dataset, fold, num_folds=5, batch_size=32):
    # Calculate size of each fold
    fold_size = len(dataset) // num_folds

    # Define train and validation indices
    val_indices = list(range(fold * fold_size, (fold + 1) * fold_size))
    train_indices = list(range(0, fold * fold_size)) + list(range((fold + 1) * fold_size, len(dataset)))

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_indices),num_workers=32)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_indices), num_workers=4)

    return train_loader, val_loader
def get_loaders(data_dir='/home/lr-2002/code/face-reg/data/lfw/', num_fold=10, batch_size=192):
    # Store loaders for each fold


    # Set a seed for reproducibility
    set_seed(42)

    # Create dataset instance
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = LFWDataset(data_dir, transform=transform)
    loaders = []
    for k in range(num_fold):
        train_loader, val_loader = get_train_val_loaders(dataset, k, batch_size=batch_size, num_folds=num_fold)
        loaders.append((train_loader, val_loader))
    return loaders

if __name__ == '__main__':
    data_dir = '/home/lr-2002/code/face-reg/data/lfw/'

    loaders = get_loaders()
    # Example usage
    for idx, (train_loader, val_loader) in enumerate(loaders):
        print(f'Fold {idx}:')
        for i, data in enumerate(val_loader):
            if i == 0:
                print(f'Val Labels: {data[1]}')
                break
        print(f'Train Loader Size: {len(train_loader.dataset)}, Val Loader Size: {len(val_loader.dataset)}')
