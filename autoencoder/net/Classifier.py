from torch import nn


class ConvClassifier(nn.Module):
  def __init__(self):
    super(ConvClassifier, self).__init__()
    self.cls = conv_classification()

  def forward(self, x):
    x = self.cls(x)
    return x

# After convolution autoencoder
def conv_classification():
  cls_net = nn.Sequential(
      nn.Flatten(),
      nn.Linear(128 * 8 * 8, 3072 // 8),
      nn.BatchNorm1d(3072 // 8),
      nn.ReLU(),
      nn.Linear(3072 // 8, 150),
      nn.BatchNorm1d(150),
      nn.ReLU(),
      nn.Linear(150, 10),
      nn.Softmax(dim=1)
  )
  return cls_net


class Classifier(nn.Module):
  def __init__(self):
    super(Classifier, self).__init__()
    self.cls = classification()

  def forward(self, x):
    x = self.cls(x)
    return x



def classification():
  cls_net = nn.Sequential(
      nn.Flatten(),
      nn.Linear(3072 // 8, 3072 // 4),
      nn.BatchNorm1d(3072 // 4),
      nn.ReLU(),
      nn.Linear(3072 // 4, 150),
      nn.BatchNorm1d(150),
      nn.ReLU(),
      nn.Linear(150, 10),
      nn.Softmax(dim=1)
  )
  return cls_net
