import numpy as np
from .strategy import Strategy
import pdb

class RandomSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, opts):
        super(RandomSampling, self).__init__(X, Y, idxs_lb, net, handler, args, opts)

    def query(self, n, modified_indicies):
        inds = np.where(self.idxs_lb==0)[0]
        chosen = inds[np.random.permutation(len(inds))][:n]

        if self.opts.mode != "clean":
            m_cnt, t_cnt = 0, 0
            for ix in range(len(chosen)):
                if chosen[ix] in modified_indicies:
                    m_cnt += 1
                t_cnt += 1
            print('  ** {}/{} clean examples **'.format(t_cnt - m_cnt, t_cnt))

        return chosen
