import numpy as np
import torch
from .strategy import Strategy

class BALDSampling(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args, opts, n_drop=10):
		super(BALDSampling, self).__init__(X, Y, idxs_lb, net, handler, args, opts)
		self.n_drop = n_drop

	def query(self, n, modified_indicies):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob_dropout_split(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled], self.n_drop)
		pb = probs.mean(0)
		entropy1 = (-pb*torch.log(pb)).sum(1)
		entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
		U = entropy2 - entropy1
		chosen = idxs_unlabeled[U.sort()[1][:n]]

		if self.opts.mode != "clean":
			m_cnt, t_cnt = 0, 0
			for ix in range(len(chosen)):
				if chosen[ix] in modified_indicies:
					m_cnt += 1
				t_cnt += 1
			print('  ** {}/{} clean examples **'.format(t_cnt - m_cnt, t_cnt))

		return chosen