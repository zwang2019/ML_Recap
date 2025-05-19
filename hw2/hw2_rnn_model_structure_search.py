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
import optuna

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

class RNNClassifier(nn.Module):
    def __init__(self, input_dim_per_frame=39, concat_nframes=11, hidden_dim=256, output_dim=41, num_layers=2, dropout_rate=0.25):
        super(RNNClassifier, self).__init__()

        self.concat_nframes = concat_nframes
        self.input_dim_per_frame = input_dim_per_frame
        self.rnn = nn.LSTM(
            input_size=input_dim_per_frame,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),   # 新增：对 hidden state 做 BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x shape: (B, concat_nframes * input_dim_per_frame)
        B = x.size(0)
        x = x.view(B, self.concat_nframes, self.input_dim_per_frame)  # reshape to (B, seq_len, input_dim_per_frame)

        output, (h_n, c_n) = self.rnn(x)  # output: (B, seq_len, hidden_dim)
        last_output = output[:, self.concat_nframes // 2, :]  # 取中间帧的输出作为分类依据

        out = self.classifier(last_output)
        return out

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
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

################################################################################################################################################
#OPTUNA

def objective(trial):

    # Hyper-parameters
    # data prarameters
    # TODO: change the value of "concat_nframes" for medium baseline
    concat_nframes = trial.suggest_int('concat_nframes', 19, 71, step=2)   # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
    train_ratio = 0.8   # the ratio of data used for training, the rest will be used for validation

    # training parameters
    seed = 1213          # random seed
    batch_size = 512        # batch size
    num_epoch = 100         # the number of training epoch
    learning_rate = 1e-4      # learning rate
    early_stop = 10
    model_path = './models/model.ckpt'  # the path where the checkpoint will be saved

    # model parameters
    # TODO: change the value of "hidden_layers" or "hidden_dim" for medium baseline
    input_dim = 39 * concat_nframes  # the input dim of the model, you should not change the value
    hidden_layers = trial.suggest_int('hidden_layers', 6, 8)          # the number of hidden layers
    hidden_dim = trial.suggest_int('hidden_dim', 450, 1024, log=True)           # the hidden dim
    dropout_rate = 0         # the dropout rate, you should not change the value
    weight_decay = 0.0

    trial_number = trial.number

    writer = SummaryWriter(log_dir=f'./RNN_model_structure_search_2/trial_{trial_number}')  # Writer of tensoboard.

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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

        trial.report(mean_val_acc, step)

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

        validation_gap = mean_val_loss - mean_train_loss
        writer.add_scalar('validation_gap', validation_gap, step)

        if early_stop_count >= early_stop:
            print(f'Model is not improving, so we halt the training session at epoch {epoch+1}')
            break

        if mean_val_loss >= 2 * mean_train_loss:
            print(f'Model is over-fitting, so we halt the training session at epoch {epoch+1}')
            break


        step += 1   # step for tensorboard

        if trial.should_prune():
            raise optuna.TrialPruned()

    '''
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
                    'best_acc': best_acc
                    })
    '''
    val_set_length = len(val_set)
    del train_set, val_set
    del train_loader, val_loader
    gc.collect()

    return best_acc/val_set_length


def run_distributed_search(storage_url, n_trials):
    # 加载或创建一个共享的 study
    study = optuna.create_study(
        study_name="RNN_model_structure_search_2",  # 共享的 study 名称
        storage=storage_url,            # 指向共享的 SQLite 文件
        load_if_exists=True,            # 如果 study 已存在，则加载它
        direction="maximize"            # 目标：最小化
    )
    # 开始优化
    study.optimize(objective, n_trials=n_trials)

    # 打印运行结束时的最佳结果
    print(f"Best value: {study.best_value}")
    print(f"Best parameters: {study.best_params}")


if __name__ == '__main__':
    storage_url = "sqlite:///RNN_model_structure_search.db"
    n_trials = 30   # Number of trials to run
    run_distributed_search(storage_url, n_trials)



# optuna-dashboard sqlite:///RNN_model_structure_search.db
# tensorboard --logdir ./RNN_model_structure_search/