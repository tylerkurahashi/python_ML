from torch import nn

# Definition of standard AutoEncoder


class AutoEncoder(nn.Module):
  def __init__(self):
    super(AutoEncoder, self).__init__()
    self.encoder = encoder()
    self.decoder = decoder()

  def forward(self, x):
    encoded_x = self.encoder(x)
    x = self.decoder(encoded_x)
    return x


def encoder():
  enc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(3072, 3072 // 2),
      nn.BatchNorm1d(3072 // 2),
      nn.ReLU(True),
      nn.Linear(3072 // 2, 3072 // 4),
      nn.BatchNorm1d(3072 // 4),
      nn.ReLU(True),
      nn.Linear(3072 // 4, 3072 // 8),
  )
  return enc


def decoder():
  dec = nn.Sequential(
      nn.Linear(3072 // 8, 3072 // 4),
      nn.BatchNorm1d(3072 // 4),
      nn.ReLU(True),
      nn.Linear(3072 // 4, 3072 // 2),
      nn.BatchNorm1d(3072 // 2),
      nn.ReLU(True),
      nn.Linear(3072 // 2, 3072),
      nn.Tanh()
  )
  return dec

# Definition of Convolutional AutoEncoder


class ConvAutoEncoder(nn.Module):
  def __init__(self):
    super(ConvAutoEncoder, self).__init__()
    self.encoder = conv_encoder()
    self.decoder = conv_decoder()
    self.output = conv_output()

  def forward(self, x):
    encoded_x = self.encoder(x)
    x = self.decoder(encoded_x)
    x = self.output(x)
    return x


def conv_encoder():
  enc = nn.Sequential(
      nn.Conv2d(3, 32, 3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(32, 64, 3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(64, 128, 3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
  )
  return enc


def conv_decoder():
  dec = nn.Sequential(
      nn.ConvTranspose2d(128, 64, 2, stride=2),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 32, 2, stride=2),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.ConvTranspose2d(32, 3, 2, stride=2),
  )
  return dec


def conv_output():
  out = nn.Sequential(
      nn.Conv2d(3, 3, 2, stride=2),
      nn.Sigmoid()
  )

  return out
