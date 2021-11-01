import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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

trainset = CustomDataset(cifar10.data, cifar10.targets, transform=transform)
# trainset = torchvision.datasets.CIFAR10(
#     root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform)

print(trainset.data.shape)

EPOCHS = 100
LR = 0.2
BATCH_SIZE = 50
losses = []

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=True)

net = AutoEncoder()
loss_function = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

save_dir = f'./ckpt/SGD_{LR}_MSE_batch{BATCH_SIZE}_{EPOCHS}epoch_unbalanced/'

os.makedirs(save_dir, exist_ok=True)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("Training Start")
for epoch in range(EPOCHS):
  start = time.time()
  running_loss = 0
  for counter, (image, _) in enumerate(trainloader, 1):
    image = image.reshape(-1, 3 * 32 * 32)

    image.to(DEVICE)

    reconstructed = net(image)

    loss = loss_function(reconstructed, image)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
  avg_loss = running_loss / counter
  losses.append(avg_loss)
  duration = time.time() - start
  print(f"Epoch {epoch + 1}  Loss:{avg_loss} {round(duration,2)} sec")
  if (epoch + 1) % 10 == 0:
    torch.save(net.state_dict(),
               save_dir + f'model_{epoch // 10}.pth'
               )

    loss_dict = {'loss': losses}
    with open(save_dir + 'loss.json', 'w') as f:
      json.dump(loss_dict, f)
