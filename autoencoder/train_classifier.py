import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Subset
import torchvision
import torchvision.transforms as transforms

from net.AutoEncoder import AutoEncoder
from dataloader.cifar import CustomCifar
from dataloader.dataset import CustomDataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar10 = CustomCifar()
cifar10.get_unbalanced_dataset()

trainvalset = CustomDataset(cifar10.data, cifar10.targets, transform=transform)
n_samples = len(trainvalset)
print(n_samples)
train_size = n_samples * 0.8

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform)

