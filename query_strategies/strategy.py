import numpy as np
from torch import nn
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from copy import deepcopy
import pdb
import resnet
import vgg
import random
from matplotlib import pyplot as plt
import os

# mlp model class
class mlpMod(nn.Module):
    def __init__(self, nClasses, dim, embSize=256):
        super(mlpMod, self).__init__()
        self.embSize = embSize
        self.dim = int(np.prod(dim))
        self.lm1 = nn.Linear(self.dim, embSize)
        self.lm2 = nn.Linear(embSize, nClasses)
    def forward(self, x):
        x = x.view(-1, self.dim)
        emb = F.relu(self.lm1(x))
        out = self.lm2(emb)
        return out, emb
    def get_embedding_dim(self):
        return self.embSize

class Strategy:
    def __init__(self, X, Y, idxs_lb, net, handler, args, opts):
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.net = net
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        if opts.model == 'mlp':
            self.ema_net = mlpMod(opts.nClasses, opts.dim, embSize=opts.nEmb)
        elif opts.model == 'resnet':
            self.ema_net = resnet.ResNet18(num_classes=opts.nClasses)
        elif opts.model == 'vgg':
            self.ema_net = vgg.VGG('VGG16', opts.nClasses)
        self.scores = None
        self.global_step = 0
        self.opts = opts
        use_cuda = torch.cuda.is_available()

    def query(self, n, modified_indicies):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def plot_sampled_images(self, x, y, epoch, nb_samples = 32, name = "visualization", 
                            dataset = "CIFAR10", savedir = "figures/"):
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
        if dataset == "CIFAR10":
            mean = [125.3, 123.0, 113.9]
            std = [63.0, 62.1, 66.7]
        
        for i, img in enumerate(x, 0):
            image = img.detach().clone()
            plt.subplot(nb_samples // 8, 8, i + 1)
            plt.axis('off')
            plt.title("{:.1f}".format(y[i].item()), fontsize = 7)
            
            for c in range(image.shape[0]):
                image[c].mul_(std[c]).add_(mean[c])
            
            plt.imshow(image.permute(1, 2, 0).to('cpu').numpy().astype('uint8'))
            if i == nb_samples - 1:
                break
        
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        savefile = os.path.join(os.path.join(savedir, "{:04d}-{}.png".format(epoch, name)))
        plt.savefig(savefile)
        plt.clf()

    def update_ema_params(self, alpha=0.999):
        alpha = min(1 - 1 / (self.global_step + 1), alpha)
        for ema_param, param in zip(self.ema_clf.parameters(), self.clf.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _finetune(self, epoch, X, Y, idxs_unlabeled, finetuning_epochs=500, prob_thresh=0.8, len_threshold=1000):
        preds = self.predict(X, Y)
        probs = self.predict_prob(X, Y).numpy()
        
        # get data for finetuning
        idxs_finetune = []
        for idx in range(len(Y)):
            if self.idxs_lb[idx]:
                idxs_finetune.append(idx)
            elif max(probs[idx]) > prob_thresh:
                idxs_finetune.append(idx)
        if len(idxs_finetune) < len_threshold:
            return -1., -1.
        loader_ft = DataLoader(self.handler(self.X[idxs_finetune], torch.Tensor(preds.numpy()[idxs_finetune]).long(), transform=self.args['transformFinetune']), shuffle=True, **self.args['loader_tr_args'])

        # print stats for the finetuning data
        print('    Finetuning on {} examples'.format(len(idxs_finetune)))
        if self.opts.mode != "clean":
            m_cnt, t_cnt = 0, 0
            for idx in idxs_finetune:
                if idx >= int(self.n_pool / (self.opts.mult + 1)):
                    m_cnt += 1
                t_cnt += 1
            print('    ** {}/{} modified examples **'.format(m_cnt, t_cnt))

        # finetune the model
        clf_acc = 0.
        eclf_acc = 0.
        optimizer = optim.Adam(self.clf.parameters(), lr = 100*self.args['lr'], weight_decay=0)
        epoch = 1
        while clf_acc < 0.99 or eclf_acc < 0.80:
            clf_acc, eclf_acc = self._train(epoch, loader_ft, optimizer)
            if epoch % 50 == 0:
                print('    [Epoch {:03d}]  OnlineModel: {:.5f}  EmaModel: {:.5f}'.format(epoch, clf_acc, eclf_acc), flush=True)
            if epoch == finetuning_epochs:
                break
            epoch += 1
        return clf_acc, eclf_acc

    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()
        self.ema_clf.train()
        clf_acc = 0.
        eclf_acc = 0.
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            # if epoch == 1:
            #     self.plot_sampled_images(x, y, epoch)
            # update online net
            x, y = x.cuda(), y.cuda()
            online_x, online_y = Variable(x), Variable(y)
            optimizer.zero_grad()
            online_out, _ = self.clf(online_x)
            loss = F.cross_entropy(online_out, online_y)
            clf_acc += torch.sum((torch.max(online_out,1)[1] == online_y).float()).data.item()
            loss.backward()
            for p in filter(lambda p: p.grad is not None, self.clf.parameters()): 
                p.grad.data.clamp_(min=-.1, max=.1)
            optimizer.step()

            # update ema net
            ema_x, ema_y = Variable(x), Variable(y)
            ema_out, _ = self.ema_clf(ema_x)
            eclf_acc += torch.sum((torch.max(ema_out,1)[1] == ema_y).float()).data.item()
            self.global_step += 1
            self.update_ema_params()

        clf_acc /= len(loader_tr.dataset.X)
        eclf_acc /= len(loader_tr.dataset.X)
        return clf_acc, eclf_acc

    def train(self, X_te, Y_te):
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        self.clf =  self.net.apply(weight_reset).cuda()
        self.ema_clf =  self.ema_net.apply(weight_reset).cuda()
        clf_optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long(), transform=self.args['transform']), shuffle=True, **self.args['loader_tr_args'])

        scores_updated = False
        epoch = 1
        max_epochs = 500
        clf_acc = 0.
        eclf_acc = 0.
        test_acc = 0.
        acc_diff = 0.
        epoch_count = 0
        criterion = False
        while clf_acc < 0.99 or eclf_acc < 0.80:
            clf_acc, eclf_acc = self._train(epoch, loader_tr, clf_optimizer)
            epoch += 1
            if epoch % 50 == 0 or (clf_acc >= 0.99 and eclf_acc >= 0.80):
                print('  [Epoch {:03d}]  OnlineModel: {:.5f}  EmaModel: {:.5f}'.format(epoch, clf_acc, eclf_acc), flush=True)
            
            # reset if not converging
            if (epoch % 50 == 0) and (clf_acc < 0.2):
                self.clf = self.net.apply(weight_reset)
                self.ema_clf = self.ema_net.apply(weight_reset)
                clf_optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)
            
            # test online net
            is_final_epoch = ((clf_acc >= 0.99 and eclf_acc >= 0.80) or epoch == max_epochs)
            if is_final_epoch == True:
                if self.opts.finetune and self.opts.alg != 'lhd':
                    idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
                    clf_acc_ft, eclf_acc_ft = self._finetune(-1, self.X, self.Y.numpy(), idxs_unlabeled)
                    print('  [After finetuning]  OnlineModel: {:.5f}  EmaModel: {:.5f}'.format(clf_acc_ft, eclf_acc_ft), flush=True)
                P = self.predict(X_te, Y_te)
                epoch_acc = 1.0 * (Y_te == P).sum().item() / len(Y_te)
                test_acc = max(test_acc, epoch_acc)
        
            # save state diff
            if clf_acc - eclf_acc < acc_diff:
                epoch_count += 1
                if epoch_count == 5:
                    criterion = True
            else:
                epoch_count = 0
            update_cond = (self.opts.alg == 'lhd' and (criterion == True or is_final_epoch == True) and scores_updated == False)
            if update_cond == True:
                if self.opts.alg == 'lhd':
                    print('  Saving State Diff')
                    idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
                    self.hidden_embeddings = self.get_hidden_embeddings_diff(self.X[idxs_unlabeled], torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long())
                    if self.opts.finetune:
                        clf_acc, eclf_acc = self._finetune(epoch, self.X, self.Y.numpy(), idxs_unlabeled)
                        print('  [Epoch {:03d}]  OnlineModel: {:.5f}  EmaModel: {:.5f}  [After finetuning]'.format(epoch, clf_acc, eclf_acc), flush=True)
                    self.scores = self.get_pseudo_loss_diff(self.X[idxs_unlabeled], torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long()).reshape((-1, 1))
                    scores_updated = True
            acc_diff = clf_acc - eclf_acc

            if epoch == max_epochs:
                break
        
        return test_acc
    
    def get_hidden_embeddings_diff(self, X, Y):
        loader_up = DataLoader(self.handler(X, Y, transform=self.args['transform']), shuffle=False, **self.args['loader_tr_args'])
        embDim = self.clf.get_embedding_dim()
        hidden_embeddings = np.zeros([len(Y), embDim])
        self.clf.eval()
        for x, _, idxs in loader_up:
            x = x.cuda()
            _, h = self.clf(Variable(x))
            _, h_ema = self.ema_clf(Variable(x))
            diff = (h_ema.detach() - h.detach()).cpu().numpy()
            for ix in range(len(idxs)):
                hidden_embeddings[idxs[ix]] = diff[ix]
        return hidden_embeddings
    
    def get_pseudo_loss_diff(self, X, Y):
        loader_up = DataLoader(self.handler(X, Y, transform=self.args['transform']), shuffle=True, **self.args['loader_tr_args'])
        L = np.zeros((len(Y), ))
        self.clf.eval()
        self.ema_clf.eval()
        with torch.no_grad():
            for x, _, idxs in loader_up:
                x = x.cuda()
                ema_out, _ = self.ema_clf(Variable(x))
                y = ema_out.max(1)[1].detach()
                ema_loss = F.cross_entropy(ema_out, y, reduction='none')

                out, _ = self.clf(Variable(x))
                loss = F.cross_entropy(out, y, reduction='none')
                L[idxs] = np.absolute(ema_loss.detach().cpu().numpy() - loss.detach().cpu().numpy())
        return L

    def predict(self, X, Y):
        if type(X) is np.ndarray:
            loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']), shuffle=False, **self.args['loader_te_args'])
        else: 
            loader_te = DataLoader(self.handler(X.numpy(), Y, transform=self.args['transformTest']), shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        P = torch.zeros(len(Y)).long()
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                P[idxs] = pred.data.cpu()
        return P

    def predict_prob(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']), shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()
        probs = torch.zeros([len(Y), len(np.unique(self.Y))])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu().data
        
        return probs

    def predict_prob_dropout_split(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for i in range(n_drop):
                for x, y, idxs in loader_te:
                    x, y = Variable(x.cuda()), Variable(y.cuda())
                    out, e1 = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu().data
            return probs

    def get_embedding(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']), shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()
        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                embedding[idxs] = e1.data.cpu()
        
        return embedding

    def get_grad_embedding(self, X, Y):
        model = self.clf
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = len(np.unique(Y))
        embedding = np.zeros([len(Y), embDim * nLab])
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']), shuffle=False, **self.args['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                cout, out = self.clf(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs,1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)