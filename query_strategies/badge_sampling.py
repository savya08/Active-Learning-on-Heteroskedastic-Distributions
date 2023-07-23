import os
import numpy as np
from .strategy import Strategy
import torch
import pdb
import argparse
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances

def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        
        if sum(D2) == 0.0:
            pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll:
            ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll

class BadgeSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, opts):
        super(BadgeSampling, self).__init__(X, Y, idxs_lb, net, handler, args, opts)

    def query(self, n, modified_indicies):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        gradEmbedding = self.get_grad_embedding(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled]).numpy()
        chosen = init_centers(gradEmbedding, n)
        
        if self.opts.mode != "clean":
            m_cnt, t_cnt = 0, 0
            for ix in range(len(chosen)):
                if idxs_unlabeled[chosen[ix]] in modified_indicies:
                    m_cnt += 1
                t_cnt += 1
            print('  ** {}/{} clean examples **'.format(t_cnt - m_cnt, t_cnt))

        return idxs_unlabeled[chosen]
        