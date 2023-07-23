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

def plot_image(image, ldiff, hdiff, embdg, name, savedir='figures/'):
    plt.title(r"$\Delta l=%.3f\   ||$\Delta h||_2=%.3f\   ||lh||_2=%.3f$"%(ldiff, hdiff, embdg))
    plt.axis('off')
    plt.imshow(image.astype('uint8'))
    savefile = os.path.join(os.path.join(savedir, name))
    plt.savefig(savefile)
    plt.clf()

class LhdSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, opts):
        super(LhdSampling, self).__init__(X, Y, idxs_lb, net, handler, args, opts)

    def query(self, n, modified_indicies):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        hidden_diff = self.hidden_embeddings
        loss_diff = self.scores
        embeddings = hidden_diff * loss_diff
        chosen = init_centers(embeddings, n)

        if self.opts.mode != "clean":
            m_cnt, t_cnt = 0, 0
            for ix in range(len(chosen)):
                if idxs_unlabeled[chosen[ix]] in modified_indicies:
                    m_cnt += 1
                t_cnt += 1
            print('  ** {}/{} clean examples **'.format(t_cnt - m_cnt, t_cnt))

        return idxs_unlabeled[chosen] 

    def analyze_embedding(self, idxs_unlabeled, hidden_diff, loss_diff, embeddings, count=20):
        hdiff_mod, hdiff_cle = 0., 0.
        ldiff_mod, ldiff_cle = 0., 0.
        embdg_mod, embdg_cle = 0., 0.
        mod_cnt, cle_cnt = 0, 0
        for i in range(min(count, len(embeddings))):
            hdiff = np.linalg.norm(hidden_diff[i])
            ldiff = loss_diff[i][0]
            embdg = np.linalg.norm(embeddings[i])
            if idxs_unlabeled[i] >= int(self.n_pool / (self.opts.mult + 1)):
                plot_image(self.X[idxs_unlabeled][i], ldiff, hdiff, embdg, name="modified-{}.png".format(i))
                hdiff_mod += hdiff
                ldiff_mod += ldiff 
                embdg_mod += embdg 
                mod_cnt += 1
            else:
                plot_image(self.X[idxs_unlabeled][i], ldiff, hdiff, embdg, name="clean-{}.png".format(i))
                hdiff_cle += hdiff
                ldiff_cle += ldiff
                embdg_cle += embdg
                cle_cnt += 1
        hdiff_mod /= mod_cnt
        ldiff_mod /= mod_cnt
        hdiff_cle /= cle_cnt
        ldiff_cle /= cle_cnt
        embdg_mod /= mod_cnt
        embdg_cle /= cle_cnt
        print('  ** Modified examples:   hdiff={:.3f} ldiff={:.3f} embdg={:.3f} **'.format(hdiff_mod, ldiff_mod, embdg_mod))
        print('  ** Clean examples:      hdiff={:.3f} ldiff={:.3f} embdg={:.3f} **'.format(hdiff_cle, ldiff_cle, embdg_cle))
        print('  ** Debug:               mod_cnt={} cle_cnt={} **'.format(mod_cnt, cle_cnt))      