import numpy as np
import pandas as pd
import math
from torch.autograd import Variable
import torch
from torch.utils.data import Dataset, DataLoader
class MyDataLoader(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = torch.tensor(self.x[index])
        y = torch.tensor(self.y[index])
        return x, y
class MyDataLoader_fortest(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = torch.tensor(self.x[index])
        return x
class MyCollator(object):

    def __init__(self, vocab, label):
        self.params = vocab
        self.label = label

    def __call__(self, batch):
        (xx, yy) = zip(*batch)
        x_len = [len(x) for x in xx]
        y_len = [len(y) for y in yy]
        batch_max_len = max([len(s) for s in xx])
        batch_data = self.params['<PAD>']*np.ones((len(xx), batch_max_len))
        batch_labels = -1*np.zeros((len(xx), batch_max_len))
        for j in range(len(xx)):
            cur_len = len(xx[j])
            batch_data[j][:cur_len] = xx[j]
            batch_labels[j][:cur_len] = yy[j]

        batch_data, batch_labels = torch.LongTensor(
            batch_data), torch.LongTensor(batch_labels)
        batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)

        return batch_data, batch_labels, x_len, y_len

class MyCollator_fortest(object):

    def __init__(self, vocab, label):
        self.params = vocab
        self.label = label

    def __call__(self, batch):
        #         print(batch)
        (xx) = batch
        x_len = [len(x) for x in xx]
        #         print(x_len)
        #         y_len = [len(y) for y in yy]
        #         batch_max_len = max([len(s) for s in xx])
        batch_data = self.params['<PAD>']*np.ones((len(xx),x_len[0]))
        #         batch_labels = -1*np.zeros((len(xx), batch_max_len))
        for j in range(len(xx)) :
            cur_len = len(xx[j])
            batch_data[j][:cur_len] = xx[j]
        #             batch_labels[j][:cur_len] = yy[j]

        batch_data = torch.LongTensor(
            batch_data)
        batch_data= Variable(batch_data)

        return batch_data, x_len