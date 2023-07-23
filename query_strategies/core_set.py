import numpy as np
import pdb
from .strategy import Strategy
from sklearn.neighbors import NearestNeighbors
import pickle
from datetime import datetime
from sklearn.metrics import pairwise_distances

class CoreSet(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, opts, tor=1e-4):
        super(CoreSet, self).__init__(X, Y, idxs_lb, net, handler, args, opts)
        self.tor = tor

    def furthest_first(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs

    def query(self, n, modified_indicies):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        lb_flag = self.idxs_lb.copy()
        embedding = self.get_embedding(self.X, self.Y)
        embedding = embedding.numpy()
        chosen = self.furthest_first(embedding[idxs_unlabeled, :], embedding[lb_flag, :], n)
        
        if self.opts.mode != "clean":
            m_cnt, t_cnt = 0, 0
            for ix in range(len(chosen)):
                if idxs_unlabeled[chosen[ix]] in modified_indicies:
                    m_cnt += 1
                t_cnt += 1
            print('  ** {}/{} clean examples **'.format(t_cnt - m_cnt, t_cnt))

        return idxs_unlabeled[chosen]