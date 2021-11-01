import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):
  def __init__(self, data, targets, transform=None, target_transform=None):
    self.data = data
    self.targets = targets
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, idx):
    img, target = np.array(self.data[idx]), np.array(self.targets[idx])

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target
