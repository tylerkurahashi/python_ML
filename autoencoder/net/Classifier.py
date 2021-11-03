from torch import nn
from torch.nn.modules.activation import Softmax
from net import AutoEncoder


class Classifier(nn.Module):
  def __init__(self):
    super(Classifier, self).__init__()
    self.cls = classification()

  def forward(self, x):
    x = self.cls(x)
    return x


def classification():
  cls_net = nn.Sequential(
      nn.Linear(3072 // 8, 3072 // 16),
      nn.ReLU(),
      nn.Linear(3072 //16, 3072 // 32),
      nn.ReLU(),
      nn.Linear(3072 // 32, 10),
      nn.Softmax()
  )
  return cls_net
