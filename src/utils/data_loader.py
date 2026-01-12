import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

def get_cifar10_loaders(data_root='./data', batch_size=64, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    # Load Full Dataset
    full_dataset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    # Stratified Split (80/20)
    targets = np.array(full_dataset.targets)
    train_idx, val_idx = train_test_split(np.arange(len(targets)), test_size=0.2, stratify=targets, random_state=42)

    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
