import pickle
from matplotlib.pyplot import axis
import numpy as np


class CustomCifar:
  def __init__(self):
    self.data = []
    self.train_data = []
    self.train_targets = []
    self.valid_data = []
    self.valid_targets = []
    self.classes = []

  def get_full_dataset(self):
    if self.data != []:
      self.data = []
      self.targets = []

    for i in range(1, 6):
      entry = unpickle(f'./data/cifar-10-batches-py/data_batch_{i}')
      self.data.append(entry["data"])
      self.targets.extend(entry["labels"])

    self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
    self.data = self.data.transpose(0, 2, 3, 1)

    self._load_meta()

  def get_unbalanced_dataset(self, train_ratio=0.8):
    num_imgs_dict = {}
    decreasing_classes = [2, 4, 9]

    if self.data != [] or self.train_data != []:
      self.train_data = []
      self.train_targets = []
      self.valid_data = []
      self.valid_targets = []

    for i in range(1, 6):
      train_remove_idx = []
      valid_remove_idx = []
      entry = unpickle(f'./data/cifar-10-batches-py/data_batch_{i}')

      for i, cls_idx in enumerate(entry['labels']):
        if cls_idx not in num_imgs_dict.keys():
          num_imgs_dict[cls_idx] = 1
          valid_remove_idx.append(i)

        else:
          # Classes to decrease the number of images to 2500
          if cls_idx in decreasing_classes:
            if num_imgs_dict[cls_idx] >= 2500:
              train_remove_idx.append(i)
              valid_remove_idx.append(i)

            elif num_imgs_dict[cls_idx] >= 2500 * train_ratio:
              train_remove_idx.append(i)

            else:
              valid_remove_idx.append(i)

          # For all other classes using 5000 images
          elif cls_idx not in decreasing_classes:
            if num_imgs_dict[cls_idx] >= 5000 * train_ratio:
              train_remove_idx.append(i)

            else:
              valid_remove_idx.append(i)

          num_imgs_dict[cls_idx] += 1

      valid_data = entry['data'].copy()
      valid_labels = entry['labels']
      valid_data = np.delete(valid_data, valid_remove_idx, axis=0)
      valid_labels = np.delete(valid_labels, valid_remove_idx, axis=0)

      train_data = np.delete(entry['data'], train_remove_idx, axis=0)
      train_labels = np.delete(entry['labels'], train_remove_idx, axis=0)

      self.train_data.append(train_data)
      self.train_targets.extend(train_labels)
      self.valid_data.append(valid_data)
      self.valid_targets.extend(valid_labels)

    # Pixel value normalization is applied
    self.train_data = np.vstack(self.train_data).reshape(-1, 3, 32, 32)
    self.train_data = self.train_data.transpose(0, 2, 3, 1)
    self.valid_data = np.vstack(self.valid_data).reshape(-1, 3, 32, 32)
    self.valid_data = self.valid_data.transpose(0, 2, 3, 1)

    print('num_imgs_dict', num_imgs_dict)
    print('train_data', self.train_data.shape)
    print('valid_data', self.valid_data.shape)

    self._load_meta()

  def _load_meta(self) -> None:
    with open('./data/cifar-10-batches-py/batches.meta', "rb") as infile:
      data = pickle.load(infile, encoding="latin1")
      self.classes = data["label_names"]


def unpickle(file):
  with open(file, 'rb') as f:
    dict = pickle.load(f, encoding='latin1')
  return dict
