from torch import nn
from AutoEncoder import AutoEncoder


class Classifier(nn.Module):
  def __init__(self):
    super(Classifier, self).__init__()
    self.ae = AutoEncoder()

  def forward(self, x):
    self.ae.encoder(x)
    
    return x

def cnn():
  conv1 = nn.Conv2d()