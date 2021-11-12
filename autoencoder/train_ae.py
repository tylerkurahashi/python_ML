import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms

from net.AutoEncoder import AutoEncoder
from dataloader.cifar import CustomCifar
from dataloader.dataset import CustomDataset

transform = transforms.Compose([
    transforms.ToTensor()
])

cifar10 = CustomCifar()
cifar10.get_unbalanced_dataset()

trainset = CustomDataset(cifar10.train_data, cifar10.train_targets, transform=transform)
# trainset = torchvision.datasets.CIFAR10(
#     root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform)

print(trainset.data.shape)

training_prefix = '20211107_3'

EPOCHS = 300
LR = 0.01
BATCH_SIZE = 100
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
save_dir = f'./ckpt/ae/{training_prefix}_SGD_{LR}_MSE_batch{BATCH_SIZE}_{EPOCHS}epoch_div8_unbalanced/'

losses = []

# To deal with the unbalnced classes here we use the weighted random sampler.
weights = []
for idx in trainset.targets:
  if idx in [2, 4, 9]:
    weights.append(0.2)
  else:
    weights.append(0.1)

weights = torch.DoubleTensor(weights)
generator = torch.Generator(device='cpu')
generator.manual_seed(1)
sampler = WeightedRandomSampler(weights,
                                len(trainset.targets),
                                replacement=True,
                                generator=generator)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=True)

net = AutoEncoder()
loss_function = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)


os.makedirs(save_dir, exist_ok=True)


print("Training Start")
for epoch in range(EPOCHS):
  start = time.time()
  running_loss = 0
  for counter, (image, _) in enumerate(trainloader, 1):
    image = image.reshape(-1, 3 * 32 * 32)

    image.to(DEVICE)
    image = image.float()

    reconstructed = net(image)

    loss = loss_function(reconstructed, image)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
  avg_loss = running_loss / counter
  losses.append(avg_loss)
  duration = time.time() - start

  for name, param in net.named_parameters():
    print(name, param.grad.abs().sum())

  print(f"Epoch {epoch + 1}  Loss:{avg_loss} {round(duration,2)} sec")
  if (epoch + 1) % 10 == 0:
    torch.save(net.state_dict(),
               save_dir + f'model_{epoch // 10}.pth'
               )

    loss_dict = {'loss': losses}
    with open(save_dir + 'metrics.json', 'w') as f:
      json.dump(loss_dict, f)
