import numpy as np
import sys
import gzip
import os
import argparse
from dataset import get_dataset, get_handler
import vgg
import resnet
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import torch
import pdb
from scipy.stats import zscore
from matplotlib import pyplot as plt
from augment import Augment, Cutout
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, CoreSet, BadgeSampling, LhdSampling, BALDSampling

parser = argparse.ArgumentParser()
parser.add_argument('--alg', help='acquisition algorithm', type=str, default='rand')
parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
parser.add_argument('--model', help='model architecture - resnet, vgg, or mlp', type=str, default='resnet')
parser.add_argument('--path', help='data path', type=str, default='data')
parser.add_argument('--save', help='logs path', type=str, default='logs')
parser.add_argument('--data', help='dataset', type=str, default='CIFAR10')
parser.add_argument('--nClasses', help='number of classes', type=int, default=10)
parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=1000)
parser.add_argument('--nStart', help='number of points to start', type=int, default=2000)
parser.add_argument('--nEnd', help = 'total number of points to query', type=int, default=12000)
parser.add_argument('--nEmb', help='number of embedding dims (mlp)', type=int, default=256)
parser.add_argument('--resume', help='whether to resume training', type=str, default='none')
parser.add_argument('--mult', help='proportion of modified examples in the training set', type=int, default=0)
parser.add_argument('--mode', help='modification mode - clean, blank, diverse, or class', type=str, default='clean')
parser.add_argument('--finetune', help='whether to finetune the classifier', action='store_true', default=False)
opts = parser.parse_args()

# parameters
NUM_INIT_LB = opts.nStart
NUM_QUERY = opts.nQuery
NUM_ROUND = int((opts.nEnd - NUM_INIT_LB) / opts.nQuery)
DATA_NAME = opts.data

args_pool = {'KMNIST':
                {'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))]),
                 'transformTest': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))]),
                 'transformFinetune': transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(28), Augment(4),
                                                transforms.ToTensor(), transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)), 
                                                Cutout(n_holes = 1, length = 14, random = True)]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 64, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
            'SVHN':
                {'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'transformTest': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'transformFinetune': transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), Augment(4),
                                                transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)), 
                                                Cutout(n_holes = 1, length = 16, random = True)]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 64, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
            'CIFAR10':
                {'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'transformTest': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'transformFinetune': transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), Augment(4),
                                                transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), 
                                                Cutout(n_holes = 1, length = 16, random = True)]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 3},
                 'loader_te_args':{'batch_size': 64, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.05, 'momentum': 0.3}}}

args = args_pool[DATA_NAME]
if not os.path.exists(opts.path):
    os.makedirs(opts.path)

X_tr, Y_tr, X_te, Y_te, modified_indicies = get_dataset(DATA_NAME, opts)
opts.dim = np.shape(X_tr)[1:]
handler = get_handler(opts.data)

args['lr'] = opts.lr

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)
print('number of labeled pool: {}'.format(NUM_INIT_LB), flush=True)
print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB), flush=True)
print('number of testing pool: {}'.format(n_test), flush=True)

# generate initial labeled pool
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_tmp = np.arange(n_pool)
np.random.shuffle(idxs_tmp)
idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

# mlp model class
class mlpMod(nn.Module):
    def __init__(self, dim, embSize=256):
        super(mlpMod, self).__init__()
        self.embSize = embSize
        self.dim = int(np.prod(dim))
        self.lm1 = nn.Linear(self.dim, embSize)
        self.lm2 = nn.Linear(embSize, opts.nClasses)
    def forward(self, x):
        x = x.view(-1, self.dim)
        emb = F.relu(self.lm1(x))
        out = self.lm2(emb)
        return out, emb
    def get_embedding_dim(self):
        return self.embSize

# load specified network
if opts.model == 'mlp':
    net = mlpMod(opts.dim, embSize=opts.nEmb)
elif opts.model == 'resnet':
    net = resnet.ResNet18(num_classes=opts.nClasses)
elif opts.model == 'vgg':
    net = vgg.VGG('VGG16', opts.nClasses)
else: 
    print('choose a valid model - mlp, resnet, or vgg', flush=True)
    raise ValueError

if type(X_tr[0]) is not np.ndarray:
    X_tr = X_tr.numpy()

# set up the specified sampler
if opts.alg == 'rand': # random sampling
    strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args, opts)
elif opts.alg == 'conf': # confidence-based sampling
    strategy = LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args, opts)
elif opts.alg == 'marg': # margin-based sampling
    strategy = MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args, opts)
elif opts.alg == 'entropy': # entropy-based sampling
    strategy = EntropySampling(X_tr, Y_tr, idxs_lb, net, handler, args, opts)
elif opts.alg == 'badge': # batch active learning by diverse gradient embeddings
    strategy = BadgeSampling(X_tr, Y_tr, idxs_lb, net, handler, args, opts)
elif opts.alg == 'coreset': # coreset sampling
    strategy = CoreSet(X_tr, Y_tr, idxs_lb, net, handler, args, opts)
elif opts.alg == 'lhd': # score-based sampling
    strategy = LhdSampling(X_tr, Y_tr, idxs_lb, net, handler, args, opts)
elif opts.alg == 'bald': # score-based sampling
    strategy = BALDSampling(X_tr, Y_tr, idxs_lb, net, handler, args, opts)
else: 
    print('Choose a valid acquisition function.', flush=True)
    raise ValueError

# print info
print(DATA_NAME, flush=True)
print(opts.alg.upper(), flush=True)
print(opts.mode.upper(), flush=True)
print()
if opts.save[-1] == '/':
    opts.save = opts.save[:-1]
save_path = '{}/{}/'.format(opts.save, opts.alg)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# round 0
strategy.train(X_te, Y_te)
P = strategy.predict(X_te, Y_te)
acc = np.zeros(NUM_ROUND+1)
acc[0] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
print('  ** Testing Accuracy: {:.5f} **\n'.format(acc[0]), flush=True)
ckpt_rd = 0

# resume training
if opts.resume != 'none':
    acc = np.load(os.path.join(save_path, 'acc.npy'))
    idxs_lb = np.load(os.path.join(save_path, 'idxs.npy'))
    strategy.idxs_lb = idxs_lb
    with open(os.path.join(save_path, 'round'), 'r') as f:
        ckpt_rd = int(f.read())
    print('  ** Training resumed from round {} **\n'.format(ckpt_rd))

for rd in range(max(ckpt_rd, 1), NUM_ROUND+1):
    print('  Round {}'.format(rd), flush=True)
    np.save(os.path.join(save_path, 'idxs.npy'), idxs_lb)
    with open(os.path.join(save_path, 'round'), 'w') as f:
        f.write(str(rd))

    # query
    output = strategy.query(NUM_QUERY, modified_indicies)
    q_idxs = output
    idxs_lb[q_idxs] = True
    print('  ** Labelled Pool Size: {} **'.format(sum(idxs_lb)), flush=True)

    # update, train, and test
    strategy.update(idxs_lb)
    acc[rd] = strategy.train(X_te, Y_te)
    print('  ** Testing Accuracy: {:.5f} **\n'.format(acc[rd]), flush=True)

    # plot round-accuracy
    plt.plot(acc[:rd], label='testing accuracy')
    plt.xlabel('round')
    plt.ylabel('accuracy')
    plt.tight_layout()
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(save_path, 'plot.png'))
    plt.clf()
    
    # save accuracies
    np.save(os.path.join(save_path, 'acc.npy'), acc)

    if sum(~strategy.idxs_lb) < opts.nQuery: 
        sys.exit('too few remaining points to query')