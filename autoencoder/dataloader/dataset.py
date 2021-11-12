import numpy as np
from PIL import Image
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
      # Converting to PIL object scales the pixel value between 0-1.
      # Which is identical to normalizing.
      img = Image.fromarray(img, 'RGB')
      img = self.transform(img)

    img = np.asarray(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target
