import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import settings

image_size = (32, 32, 3)
height, width, channels = image_size

def get_dataloader(train:bool, transform:nn.Module, batch_size):
    dataset = torchvision.datasets.CIFAR10(
        root=settings.DATA_DIR,        
        train=False,
        download=True,
        transform=transform,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

def get_test_dataloader(batch_size):
    transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    return get_dataloader(train=False, transform=transform, batch_size=batch_size)


def get_train_dataloader(batch_size):
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.Resize((height, width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    return get_dataloader(train=True, transform=transform, batch_size=batch_size)
