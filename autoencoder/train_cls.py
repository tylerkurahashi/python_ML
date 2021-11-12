import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms

from net import Classifier, ConvClassifier
from net import AutoEncoder, ConvAutoEncoder
from dataloader import CustomCifar
from dataloader import CustomDataset
from utils import cls_train

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=5),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar10 = CustomCifar()
cifar10.get_unbalanced_dataset(train_ratio=0.8)

trainset = CustomDataset(
    cifar10.train_data, cifar10.train_targets, transform=train_transform)
validset = CustomDataset(
    cifar10.valid_data, cifar10.valid_targets, transform=valid_transform)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=valid_transform)

training_prefix = '20211111_2_conv'

EPOCHS = 300
LR = 1e-3
BATCH_SIZE = 128
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ae_model_path = f'./ckpt/ae/20211111_2_conv_da_bn_Adam_0.001_MSE_batch128_300epoch_div8_unbalanced/model_30.pth'
save_dir = f'./ckpt/cls/{training_prefix}_Adam_{LR}_CEL_batch{BATCH_SIZE}_{EPOCHS}epoch_div8_unbalanced/'

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
    trainset, batch_size=BATCH_SIZE, shuffle=False, sampler=sampler, num_workers=2, drop_last=False)
validloader = torch.utils.data.DataLoader(
    validset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=False)

# Select the model to use.
########################

# ae = AutoEncoder()
ae = ConvAutoEncoder()

# cls_net = Classifier()
cls_net = ConvClassifier()

########################

ae.load_state_dict(torch.load(ae_model_path))
ae.eval()


loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    cls_net.parameters(),
    lr=LR,
    weight_decay=1e-5)

os.makedirs(save_dir, exist_ok=True)

train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []

print("Training Start")

for epoch in range(EPOCHS):
  start = time.time()

  # Iterate through training dataset
  train_avg_loss, train_acc = cls_train(
      dataloader=trainloader,
      ae_model=ae,
      cls_model=cls_net,
      loss_func=loss_function,
      optim=optimizer,
      mode='train')

  valid_avg_loss, valid_acc = cls_train(
      dataloader=validloader,
      ae_model=ae,
      cls_model=cls_net,
      loss_func=loss_function,
      mode='valid')

  # Metrics to save
  train_losses.append(train_avg_loss)
  train_accuracies.append(train_acc)
  valid_losses.append(valid_avg_loss)
  valid_accuracies.append(valid_acc)

  duration = time.time() - start

  print(f"""
    Epoch {epoch + 1}:
    Train Loss:{train_avg_loss} valid Loss:{valid_avg_loss}
    Train Acc:{train_acc} valid Acc:{valid_acc}
    Training duration: {round(duration,2)}sec
    """)

  if (epoch + 1) % 10 == 0:
    torch.save(
        cls_net.state_dict(),
        save_dir + f'model_{(epoch + 1) // 10}.pth'
    )

    saving_metrics = {
        'train_loss': train_losses,
        'train_acc': train_accuracies,
        'valid_loss': valid_losses,
        'valid_acc': valid_accuracies,
        # 'test_loss': test_losses,
        # 'test_acc': test_accuracies
    }
    with open(save_dir + 'metrics.json', 'w') as f:
      json.dump(saving_metrics, f)
