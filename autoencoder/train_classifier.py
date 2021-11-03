import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Subset
import torchvision
import torchvision.transforms as transforms

from net import Classifier
from net import AutoEncoder
from dataloader import CustomCifar
from dataloader import CustomDataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar10 = CustomCifar()
cifar10.get_unbalanced_dataset()

trainset = CustomDataset(cifar10.data, cifar10.targets, transform=transform)
# n_samples = len(trainvalset)
# print(n_samples)
# train_size = n_samples * 0.8

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform)

print(trainset.data.shape)

EPOCHS = 100
LR = 0.1
BATCH_SIZE = 100
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ae_model_path = f'./ckpt/SGD_0.1_MSE_batch100_100epoch_div8_unbalanced/model_9.pth'
save_dir = f'./ckpt/test_cls_SGD_{LR}_MSE_batch{BATCH_SIZE}_{EPOCHS}epoch_div8_unbalanced/'

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=True)

ae = AutoEncoder()
ae.load_state_dict(torch.load(ae_model_path))
ae.eval()
net = Classifier()
loss_function = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

os.makedirs(save_dir, exist_ok=True)

print("Training Start")


for epoch in range(EPOCHS):
  start = time.time()
  train_correct_pred = 0
  test_correct_pred = 0
  train_running_loss = 0
  test_running_loss = 0

  # Iterate through training dataset
  for train_counter, (image, label) in enumerate(trainloader, 1):
    image = image.reshape(-1, 3 * 32 * 32)
    onehot_label = nn.functional.one_hot(label, num_classes=10)
    onehot_label = onehot_label.to(torch.float32)

    image.to(DEVICE)
    onehot_label.to(DEVICE)

    embedding = ae.encoder(image)
    pred_class = net(embedding)

    loss = loss_function(pred_class, onehot_label)

    # Count correct prediction
    pred_class = np.argmax(pred_class.detach().numpy(), axis=1)
    row_label = label.detach().numpy()
    train_correct_pred += (pred_class == row_label).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_running_loss += loss.item()

  # Iterate through test dataset
  for test_counter, (image, label) in enumerate(testloader, 1):
    image = image.reshape(-1, 3 * 32 * 32)
    onehot_label = nn.functional.one_hot(label, num_classes=10)
    onehot_label = onehot_label.to(torch.float32)

    image.to(DEVICE)
    onehot_label.to(DEVICE)

    embedding = ae.encoder(image)
    pred_class = net(embedding)

    loss = loss_function(pred_class, onehot_label)

    # Count correct prediction
    pred_class = np.argmax(pred_class.detach().numpy(), axis=1)
    row_label = label.detach().numpy()
    test_correct_pred += (pred_class == row_label).sum()

    test_running_loss += loss.item()

  train_avg_loss = train_running_loss / train_counter
  test_avg_loss = test_running_loss / test_counter
  train_acc = train_correct_pred / trainset.data.shape[0]
  test_acc = test_correct_pred / testset.data.shape[0]
  train_losses.append(train_avg_loss)
  test_losses.append(test_avg_loss)
  train_accuracies.append(train_acc)
  test_accuracies.append(test_acc)

  duration = time.time() - start

  print(f"""
    Epoch {epoch + 1}:
    Train Loss:{train_avg_loss} Test Loss:{test_avg_loss}
    Train Acc:{train_acc} Test Acc:{test_acc}
    Training duration: {round(duration,2)}sec
    """)

  if (epoch + 1) % 10 == 0:
    torch.save(net.state_dict(),
               save_dir + f'model_{epoch // 10}.pth'
               )

    saving_dict = {
        'train_loss': train_losses,
        'test_loss': test_losses,
        'train_acc': train_accuracies,
        'test_acc': test_accuracies
    }
    with open(save_dir + 'loss.json', 'w') as f:
      json.dump(saving_dict, f)
