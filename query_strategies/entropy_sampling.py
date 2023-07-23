import numpy as np
import torch
from .strategy import Strategy

class EntropySampling(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args, opts):
		super(EntropySampling, self).__init__(X, Y, idxs_lb, net, handler, args, opts)

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled])
		log_probs = torch.log(probs)
		U = (probs*log_probs).sum(1)

		m_cnt, t_cnt = 0, 0
		for ix in range(len(chosen)):
			if chosen[ix] >= int(self.n_pool / (self.opts.mult + 1)):
				m_cnt += 1
			t_cnt += 1
		print('  ** {}/{} clean examples **'.format(t_cnt - m_cnt, t_cnt))
        
		return idxs_unlabeled[U.sort()[1][:n]]
