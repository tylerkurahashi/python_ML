import pickle
from matplotlib.pyplot import axis
import numpy as np


def unpickle(file):
  with open(file, 'rb') as f:
    dict = pickle.load(f, encoding='latin1')
  return dict


class CustomCifar:
  def __init__(self):
    self.data = []
    self.targets = []
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

  def get_unbalanced_dataset(self):
    num_imgs_dict = {}

    if self.data != []:
      self.data = []
      self.targets = []

    for i in range(1, 6):
      remove_idx = []
      entry = unpickle(f'./data/cifar-10-batches-py/data_batch_{i}')
      for i, cls_idx in enumerate(entry['labels']):
        if cls_idx not in num_imgs_dict.keys():
          num_imgs_dict[cls_idx] = 1
        else:
          if cls_idx in [2, 4, 9] and num_imgs_dict[cls_idx] >= 2500:
            remove_idx.append(i)
          else:
            num_imgs_dict[cls_idx] += 1

      # print('removing', remove_idx)
      data = np.delete(entry['data'], remove_idx, axis=0)
      labels = np.delete(entry['labels'], remove_idx, axis=0)

      self.data.append(data)
      self.targets.extend(labels)

    print(num_imgs_dict)

    print(np.array(self.data).shape)
    self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
    self.data = self.data.transpose(0, 2, 3, 1)

    self._load_meta()

  def _load_meta(self) -> None:
    with open('./data/cifar-10-batches-py/batches.meta', "rb") as infile:
      data = pickle.load(infile, encoding="latin1")
      self.classes = data["label_names"]
