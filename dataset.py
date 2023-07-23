import numpy as np
import pdb
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import random

def get_colored_examples(X, Y, sz, num_k=100, nl=3):
    X_mod = np.zeros((sz, X.shape[1], X.shape[2], X.shape[3]), dtype = np.uint8)
    Y_mod = np.zeros((sz, ))
    rand_colors = []
    rand_labs = []

    for k in range(0, num_k):
        if X.shape[1] == 3:
            rand_colors.append((255*torch.rand(X.shape[1], 1, 1).repeat(1, X.shape[2], X.shape[3])).numpy().astype('uint8'))
        else:
            rand_colors.append((255*torch.rand(1, 1, X.shape[3]).repeat(X.shape[1], X.shape[2], 1)).numpy().astype('uint8'))
        rand_labs.append(random.sample(list(np.unique(Y)), k=nl))
    
    for j in range(X_mod.shape[0]):
        rand_k = random.randint(0, num_k-1)
        col = rand_colors[rand_k]
        labs = rand_labs[rand_k]
        X_mod[j] += col
        Y_mod[j] += random.sample(labs, 1)[0]
    return X_mod, Y_mod

def get_random_class_examples(X, Y, sz, random_class_examples_idxs, k=100):
    X_mod = np.zeros((sz, X.shape[1], X.shape[2], X.shape[3]), dtype = np.uint8)
    Y_mod = np.zeros((sz, ))
    for j in range(X_mod.shape[0]):
        X_mod[j] += X[random.choice(random_class_examples_idxs[:k])]
        Y_mod[j] += np.random.randint(0, 10)
    return X_mod, Y_mod

def add_modified_examples(X, Y, mode, mult, same_label=10):
    modified_indicies = np.arange(start=len(Y), stop=len(Y) * (mult + 1))
    if mode == 'blank':
        sz = X.shape[0]*mult
        X_mod = np.zeros((sz, X.shape[1], X.shape[2], X.shape[3]), dtype = np.uint8)
        Y_mod = np.random.randint(0, 10, (sz, ))
    elif mode == 'diverse':
        sz = X.shape[0]*mult
        X_mod, Y_mod = get_colored_examples(X, Y, sz)
    elif mode == 'class':
        random_class_examples_idxs = np.where(Y==1)[0]
        modified_indicies = np.concatenate((modified_indicies, np.where(Y == 1)[0]), axis=0)
        Y[random_class_examples_idxs] = np.random.randint(0, 10, size=len(random_class_examples_idxs)).astype(float)
        sz = X.shape[0] * mult
        X_mod, Y_mod = get_random_class_examples(X, Y, sz, random_class_examples_idxs)
    elif mode == 'clean':
        return X, Y, None
    else:
        print("Enter a valid mode for modified examples")
        exit()
    X_res = np.concatenate((X, X_mod), axis=0)
    Y_res = np.concatenate((Y, Y_mod), axis=0)
    return X_res, Y_res, modified_indicies

def get_dataset(name, opts):
    if name == 'CIFAR10':
        return get_CIFAR10(opts)
    elif name == 'SVHN':
        return get_SVHN(opts)
    elif name == 'KMNIST':
        return get_KMNIST(opts)

def get_CIFAR10(opts):
    modified_indicies = None
    data_tr = datasets.CIFAR10(opts.path + '/CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10(opts.path + '/CIFAR10', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = np.array(data_tr.targets)
    X_tr, Y_tr, modified_indicies = add_modified_examples(X_tr, Y_tr, opts.mode, opts.mult)
    Y_tr = torch.from_numpy(Y_tr)
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te, modified_indicies

def get_SVHN(opts):
    modified_indicies = None
    data_tr = datasets.SVHN(opts.path + '/SVHN', split='train', download=True)
    data_te = datasets.SVHN(opts.path +'/SVHN', split='test', download=True)
    X_tr = data_tr.data
    Y_tr = np.array(data_tr.labels)
    X_tr, Y_tr, modified_indicies = add_modified_examples(X_tr, Y_tr, opts.mode, opts.mult)
    Y_tr = torch.from_numpy(Y_tr)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)
    return X_tr, Y_tr, X_te, Y_te, modified_indicies

def get_KMNIST(opts):
    modified_indicies = None
    data_tr = datasets.KMNIST(opts.path + '/KMNIST', train=True, download=True)
    data_te = datasets.KMNIST(opts.path + '/KMNIST', train=False, download=True)
    X_tr = data_tr.data
    X_tr = X_tr.unsqueeze(1).repeat(1, 3, 1, 1)
    Y_tr = np.array(data_tr.targets)
    X_tr, Y_tr, modified_indicies = add_modified_examples(X_tr, Y_tr, opts.mode, opts.mult)
    Y_tr = torch.from_numpy(Y_tr)
    X_te = data_te.data
    X_te = X_te.unsqueeze(1).repeat(1, 3, 1, 1)
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te, modified_indicies

def get_handler(name):
    if name == 'CIFAR10':
        return DataHandler1
    elif name == 'SVHN':
        return DataHandler2
    elif name == 'KMNIST':
        return DataHandler2
    else:
        print("Enter a valid dataset")
        exit()

class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)