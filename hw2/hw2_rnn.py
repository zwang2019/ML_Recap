# library import

import os
import gc

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import math

import random
from tqdm import tqdm

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter


################################################################################################################################################

# Utility Functions

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_feat(path):
    feat = torch.load(path)
    return feat

def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]
    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)

def concat_feat(x, concat_n):
    assert concat_n % 2 == 1 # n must be odd
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n)
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2) # concat_n, seq_len, feature_dim
    mid = (concat_n // 2)
    for r_idx in range(1, mid+1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)

def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, random_seed=1213):
    class_num = 41 # NOTE: pre-computed should not need change

    if split == 'train' or split == 'val':
        mode = 'train'
    elif split == 'test':
        mode = 'test'
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    label_dict = {}
    if mode == 'train':
        for line in open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines():
            line = line.strip('\n').split(' ')
            label_dict[line[0]] = [int(p) for p in line[1:]]

        # split training and validation data
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(random_seed)
        random.shuffle(usage_list)
        train_len = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:train_len] if split == 'train' else usage_list[train_len:]

    elif mode == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode == 'train':
        y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode == 'train':
          label = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode == 'train':
          y[idx: idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode == 'train':
      y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode == 'train':
      print(y.shape)
      return X, y
    else:
      return X


# Dataset

class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

################################################################################################################################################

# Model
# Feel free to modify the structure of the model.
class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.15):
        super(BasicBlock, self).__init__()

        # TODO: apply batch normalization and dropout for strong baseline.
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html (batch normalization)
        #       https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html (dropout)
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256, dropout_rate=0.25):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim, dropout_rate),
            *[BasicBlock(hidden_dim, hidden_dim, dropout_rate) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class RNNClassifier(nn.Module):
    def __init__(self, input_dim_per_frame=39, concat_nframes=11, hidden_dim=256, output_dim=41, num_layers=2, dropout_rate=0.25):
        super(RNNClassifier, self).__init__()

        self.concat_nframes = concat_nframes
        self.input_dim_per_frame = input_dim_per_frame
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim

        self.rnn = nn.LSTM(
            input_size=self.input_dim_per_frame,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        '''
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(2 * self.hidden_size),   # 新增：对 hidden state 做 BatchNorm
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(2 * self.hidden_size, self.output_dim)
        )
        '''

        self.fc = nn.Sequential(
            # 修改成 2 * self.hidden_size 的原因是因为LSTM()中的bidirectional设置为了True，这表示使用Bi（双向）LSTM模型，所以需要修改输入维度以匹配
            BasicBlock(2 * self.hidden_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.output_dim)
        )

    def forward(self, x):
        # x.shape: (batch_size, seq_len, RNN_input_size)
        x, _ = self.rnn(x)  # => (batch_size, seq_len, RNN_hidden_size)
        x = x[:, -1]  # => (batch_size, RNN_hidden_size)
        x = self.fc(x)  # => (batch_size, labels)
        return x

################################################################################################################################################

# Hyper-parameters
# data prarameters
# TODO: change the value of "concat_nframes" for medium baseline
concat_nframes = 63   # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.8   # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 1213          # random seed
batch_size = 512        # batch size
num_epoch = 300         # the number of training epoch
learning_rate = 1e-4      # learning rate
early_stop = 30
model_path = './models/model.ckpt'  # the path where the checkpoint will be saved

# model parameters
# TODO: change the value of "hidden_layers" or "hidden_dim" for medium baseline
input_dim = 39 * concat_nframes  # the input dim of the model, you should not change the value
hidden_layers = 7          # the number of hidden layers
hidden_dim = 760           # the hidden dim
dropout_rate = 0.35         # the dropout rate, you should not change the value
weight_decay = 7e-5


writer = SummaryWriter(log_dir=f'./RNN_model_search/try_1')  # Writer of tensoboard.

################################################################################################################################################

# Dataloader
same_seeds(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')

# preprocess data
train_X, train_y = preprocess_data(split='train', feat_dir='./data/feat', phone_path='./data', concat_nframes=concat_nframes, train_ratio=train_ratio, random_seed=seed)
val_X, val_y = preprocess_data(split='val', feat_dir='./data/feat', phone_path='./data', concat_nframes=concat_nframes, train_ratio=train_ratio, random_seed=seed)

# get dataset
train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)

# remove raw feature to save memory
del train_X, train_y, val_X, val_y
gc.collect()

# get dataloader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

################################################################################################################################################

# Training
# create model, define a loss function, and optimizer


# model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim, dropout_rate=dropout_rate).to(device)
model = RNNClassifier(
    input_dim_per_frame=39,
    concat_nframes=concat_nframes,
    hidden_dim=hidden_dim,
    output_dim=41,
    num_layers=hidden_layers,
    dropout_rate=dropout_rate
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

best_acc = 0.0
step = 0
early_stop_count = 0

for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train() # set the model to training mode
    for i, batch in enumerate(tqdm(train_loader)):
        features, labels = batch
        features = features.to(device)
        features = features.view(-1, concat_nframes, 39).to(device)  # feature.shape: (batch_size, seq_len, input_size)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        train_acc += (train_pred.detach() == labels.detach()).sum().item()
        train_loss += loss.item()

    # log the training loss and accuracy to tensorboard
    mean_train_loss = train_loss/len(train_loader)
    mean_train_acc = train_acc/len(train_set)
    writer.add_scalar('train/loss', mean_train_loss, step)
    writer.add_scalar('train/acc', mean_train_acc, step)

    # validation
    model.eval() # set the model to evaluation mode
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            features, labels = batch
            features = features.to(device)
            features = features.view(-1, concat_nframes, 39).to(device)
            labels = labels.to(device)
            outputs = model(features)

            loss = criterion(outputs, labels)

            _, val_pred = torch.max(outputs, 1)
            val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
            val_loss += loss.item()

    # log the validation loss and accuracy to tensorboard
    mean_val_loss = val_loss/len(val_loader)
    mean_val_acc = val_acc/len(val_set)
    writer.add_scalar('val/loss', mean_val_loss, step)
    writer.add_scalar('val/acc', mean_val_acc, step)

    print(f'[{epoch+1:03d}/{num_epoch:03d}] Train Acc: {mean_train_acc:3.5f} Loss: {mean_train_loss:3.5f} | Val Acc: {mean_val_acc:3.5f} loss: {mean_val_loss:3.5f}')

    # if the model improves, save a checkpoint at this epoch
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), model_path)
        print(f'saving model with acc {best_acc/len(val_set):.5f}')
        writer.add_scalar('val/best_acc', best_acc/len(val_set), step)
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= early_stop:
        print(f'Model is not improving, so we halt the training session at epoch {epoch+1}')
        break

    step += 1   # step for tensorboard

# log the hyper-parameters
writer.add_hparams(
                {
                'concat_nframes': concat_nframes,
                'train_ratio': train_ratio,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_epoch': num_epoch,
                'early_stop': early_stop,
                'hidden_layers': hidden_layers,
                'hidden_dim': hidden_dim,
                'weight_decay': weight_decay,
                'dropout': dropout_rate,
                },
                {
                'best_acc': best_acc/len(val_set)
                })

del train_set, val_set
del train_loader, val_loader
gc.collect()

################################################################################################################################################

# Testing
# Create a testing dataset, and load model from the saved checkpoint.

# load data
test_X = preprocess_data(split='test', feat_dir='./data/feat', phone_path='./data', concat_nframes=concat_nframes)
test_set = LibriDataset(test_X, None)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# load model
# model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
model = RNNClassifier(
    input_dim_per_frame=39,
    concat_nframes=concat_nframes,
    hidden_dim=hidden_dim,
    output_dim=41,
    num_layers=hidden_layers,
    dropout_rate=dropout_rate
).to(device)
model.load_state_dict(torch.load(model_path))

# Make prediction.

pred = np.array([], dtype=np.int32)
model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        features = batch
        features = features.to(device)
        features = features.view(-1, concat_nframes, 39).to(device)

        outputs = model(features)

        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

# Save the prediction to a file
with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(pred):
        f.write('{},{}\n'.format(i, y))