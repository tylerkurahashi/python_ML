import numpy as np
import torch
import torch.nn as nn

# mode argument must be set to 'train' or some other value (ex. None)


def cls_train(dataloader, ae_model, cls_model, loss_func=None, optim=None, mode: str = 'train'):
  running_loss = 0
  correct_pred = 0

  ae_model = ae_model.eval()
  if mode == 'train':
    cls_model = cls_model.train()
  else:
    cls_model = cls_model.eval()

  DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  for counter, (img, label) in enumerate(dataloader):

    onehot_label = nn.functional.one_hot(label, num_classes=10)
    onehot_label = onehot_label.to(torch.float32)

    img = img.float()
    img.to(DEVICE)
    onehot_label.to(DEVICE)

    embedding = ae_model.encoder(img)
    pred_class = cls_model(embedding)

    if mode == 'train' or mode == 'valid':
      loss = loss_func(pred_class, onehot_label)

      if mode == 'train':
        optim.zero_grad()
        loss.backward()
        optim.step()

      running_loss += loss.item()

    # Count correct prediction
    pred_class = np.argmax(pred_class.detach().numpy(), axis=1)
    raw_label = label.detach().numpy()
    correct_pred += (pred_class == raw_label).sum()

  avg_loss = round(running_loss / counter, 4)
  acc = round(correct_pred / len(dataloader) / dataloader.batch_size, 4)

  return avg_loss, acc


def conv_ae_train(dataloader, ae_model, loss_func, optim=None, type='stacking', mode: str = 'train'):
  running_loss = 0
  if mode != 'train':
    ae_model = ae_model.eval()

  DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  for counter, (img, _) in enumerate(dataloader):

    img = img.float()
    img.to(DEVICE)

    embedding = ae_model(img)

    if type == 'stacking':
      embedding = embedding.reshape(-1, 3, 32, 32)

    loss = loss_func(embedding, img)

    if mode == 'train':
      optim.zero_grad()
      loss.backward()
      optim.step()

    running_loss += loss.item()

  avg_loss = round(running_loss / counter, 4)

  return avg_loss
