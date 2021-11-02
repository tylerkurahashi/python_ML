from torch import nn


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
      nn.Linear(3072, 3072 // 2),
      nn.ReLU(True),
      nn.Linear(3072 // 2, 3072 // 4),
      nn.ReLU(True),
      nn.Linear(3072 // 4, 3072 // 8)
  )
  return enc


def decoder():
  dec = nn.Sequential(
      nn.Linear(3072 // 8, 3072 // 4),
      nn.ReLU(True),
      nn.Linear(3072 // 4, 3072 // 2),
      nn.ReLU(True),
      nn.Linear(3072 // 2, 3072),
      nn.Tanh()
  )
  return dec
