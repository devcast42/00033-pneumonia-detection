import os
from torchvision import datasets
from torch.utils.data import DataLoader
from .preprocessing import get_train_transforms, get_val_transforms

def get_datasets(data_dir):
    """
    Load datasets from the directory structure.
    Expected structure:
    data_dir/
        train/
            NORMAL/
            PNEUMONIA/
        val/
            NORMAL/
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # Verify directories exist
    for d in [train_dir, val_dir, test_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(f"Directory not found: {d}")

    train_dataset = datasets.ImageFolder(train_dir, transform=get_train_transforms())
    val_dataset = datasets.ImageFolder(val_dir, transform=get_val_transforms())
    test_dataset = datasets.ImageFolder(test_dir, transform=get_val_transforms())

    return train_dataset, val_dataset, test_dataset

def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Create DataLoaders for train, val, test.
    """
    train_dataset, val_dataset, test_dataset = get_datasets(data_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx
