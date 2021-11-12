import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.transforms as transforms

from net.AutoEncoder import AutoEncoder, ConvAutoEncoder
from dataloader.cifar import CustomCifar
from dataloader.dataset import CustomDataset
from utils import conv_ae_train

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=5),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar10 = CustomCifar()
cifar10.get_unbalanced_dataset()

trainset = CustomDataset(
    cifar10.train_data, cifar10.train_targets, transform=transform)

training_prefix = '20211111_2_conv_da_bn'

EPOCHS = 300
LR = 0.001
BATCH_SIZE = 128
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

save_dir = f'./ckpt/ae/{training_prefix}_Adam_{LR}_MSE_batch{BATCH_SIZE}_{EPOCHS}epoch_div8_unbalanced/'


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
sampler = WeightedRandomSampler(
    weights,
    len(trainset.targets),
    replacement=True,
    generator=generator)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    sampler=sampler,
    num_workers=2,
    drop_last=False)

# net = AutoEncoder()
net = ConvAutoEncoder()
loss_function = nn.MSELoss()
optimizer = optim.Adam(
    net.parameters(),
    lr=LR,
    weight_decay=1e-5)

os.makedirs(save_dir, exist_ok=True)

losses = []

print("Training Start")

for epoch in range(EPOCHS):
  start = time.time()

  avg_loss = conv_ae_train(
      trainloader,
      ae_model=net,
      loss_func=loss_function,
      optim=optimizer,
      # type='stacking',
      type='conv',
      mode='train')

  duration = time.time() - start

  for name, param in net.named_parameters():
    print(name, param.grad.abs().sum())

  losses.append(avg_loss)

  print(f"Epoch {epoch + 1}  Loss:{avg_loss} {round(duration,2)} sec")

  if (epoch + 1) % 10 == 0:
    torch.save(
        net.state_dict(),
        save_dir + f'model_{(epoch + 1) // 10}.pth'
    )

    loss_dict = {'loss': losses}
    with open(save_dir + 'metrics.json', 'w') as f:
      json.dump(loss_dict, f)
