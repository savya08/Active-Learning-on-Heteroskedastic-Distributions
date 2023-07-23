import numpy as np
import torch
from .strategy import Strategy
import pdb

class LeastConfidence(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, opts):
        super(LeastConfidence, self).__init__(X, Y, idxs_lb, net, handler, args, opts)

    def query(self, n, modified_indicies):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], np.asarray(self.Y)[idxs_unlabeled])
        U = probs.max(1)[0]
        chosen = idxs_unlabeled[U.sort()[1][:n]]
        
        if self.opts.mode != "clean":
            m_cnt, t_cnt = 0, 0
            for ix in range(len(chosen)):
                if chosen[ix] in modified_indicies:
                    m_cnt += 1
                t_cnt += 1
            print('  ** {}/{} clean examples **'.format(t_cnt - m_cnt, t_cnt))

        return chosen
