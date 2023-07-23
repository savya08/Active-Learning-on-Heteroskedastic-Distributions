# Run python3 two_moons.py

import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch
import random
from math import ceil
from query_strategies import LeastConfidence
from sklearn import datasets
from sklearn.metrics import pairwise_distances
import pdb
from scipy import stats

def furthest_first(X, X_set, n):
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


def diversity_query(x, data_size, idxs_lb, n, net):
    idxs_unlabeled = np.arange(data_size)[~idxs_lb]
    with torch.no_grad():
        embedding = net.get_embedding(x)
    embedding = embedding.numpy()
    chosen = furthest_first(embedding[idxs_unlabeled, :], embedding[idxs_lb, :], n)
    return idxs_unlabeled[chosen]

def confidence_query(x, data_size, idxs_lb, n):
    idxs_unlabeled = np.arange(data_size)[~idxs_lb]
    with torch.no_grad():
        probs = net(torch.tensor(x[idxs_unlabeled]))
        probs = torch.max(probs, dim=1)[0]
    return idxs_unlabeled[probs.argsort()[:n]]

def update_ema_params(net, ema_net, global_step, alpha=0.999):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_net.parameters(), net.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def get_hidden_embeddings_diff(net, ema_net, X):
    net.eval()
    ema_net.eval()
    h = net.get_embedding(X)
    h_ema = ema_net.get_embedding(X)
    diff = (h_ema.detach() - h.detach()).cpu().numpy()
    return diff

def get_pseudo_loss_diff(net, ema_net, X):
    net.eval()
    ema_net.eval()
    with torch.no_grad():
        ema_out = ema_net(X)
        y = ema_out.max(1)[1].detach()
        ema_loss = F.cross_entropy(ema_out, y, reduction='none')
        out = net(X)
        loss = F.cross_entropy(out, y, reduction='none')
        loss_diff = np.absolute(ema_loss.detach().cpu().numpy() - loss.detach().cpu().numpy())
    return loss_diff


class mlpMod(nn.Module):
    def __init__(self, in_dim, out_dim, embSize=512):
        super(mlpMod, self).__init__()
        self.embSize = embSize
        self.in_dim = int(np.prod(in_dim))
        self.lm1 = nn.Linear(self.in_dim, embSize)
        self.lm2 = nn.Linear(embSize, embSize)
        self.lm3 = nn.Linear(embSize, out_dim)

    def forward(self, x):
        x = x.view(-1, self.in_dim)
        x = F.relu(self.lm1(x))
        x = F.relu(self.lm2(x))
        out = self.lm3(x)
        return out

    def get_embedding_dim(self):
        return self.embSize

    def get_embedding(self, x):
        x = x.view(-1, self.in_dim)
        x = F.relu(self.lm1(x))
        emb = F.relu(self.lm2(x))
        return emb

def create_moon_data(data_size=12000, num_class=4, verbose=False):
    x_1, y_1 = datasets.make_moons(n_samples=int(12000/2), noise=0.1, random_state=123)
    x_2, y_2 = datasets.make_moons(n_samples=int(12000/2), noise=0.1, random_state=123)
    y_2 = y_2 + 2
    x_2 = x_2 + np.array([4, 0])
    x = np.concatenate((x_1, x_2), axis=0)
    y = np.concatenate((y_1, y_2), axis=0)
    noisy_mask = (y == 0)
    noisy_idxs = np.arange(len(y))[y == 0]
    y[noisy_mask] = np.random.randint(num_class, size=sum(y==0))
    noisy_x = x[noisy_mask]
    noisy_y = y[noisy_mask]
    for i in range(6):
        x = np.concatenate((x, np.copy(noisy_x)))
        y = np.concatenate((y, np.copy(noisy_y)))
    noisy_idxs = np.concatenate((noisy_idxs, np.arange(len(noisy_mask), len(y))), axis=0)
    print(noisy_idxs)
    if verbose:
        print(f"X Shape: {x.shape}")
        print(f"y Shape: {y.shape}")
        plt.clf()
        plt.scatter(x=x[:, 0], y=x[:, 1], c=y)
        plt.title("Total Dataset")
        plt.xlim(-2., 7.)
        plt.ylim(-1., 1.5)
        plt.savefig("figures/Total_Dataset.png")
        plt.show()

    x_1, y_1 = datasets.make_moons(n_samples=int(12000/2), noise=0.1, random_state=123)
    x_2, y_2 = datasets.make_moons(n_samples=int(12000/2), noise=0.1, random_state=123)
    y_2 = y_2 + 2
    x_2 = x_2 + np.array([4, 0])
    test_x = np.concatenate((x_1, x_2), axis=0)
    test_y = np.concatenate((y_1, y_2), axis=0)
    return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.long), torch.tensor(test_x, dtype=torch.float), torch.tensor(test_y, dtype=torch.long), noisy_idxs


def train(net, ema_net, x, y, batch_size, epochs, optimizer, x_unlabeled, global_step):
    print(x.size())
    criterion = nn.CrossEntropyLoss()
    num_batches = ceil(len(x) / batch_size)
    shuffled_indices = np.arange(len(x))
    scores_updated = False
    update_criteria = False
    epoch_count = 0
    acc_diff = 0.
    for epoch in range(epochs):
        random.shuffle(shuffled_indices)
        running_loss = 0.0
        net_acc = 0.
        ema_net_acc = 0.
        for batch_idx in range(num_batches):
            inputs = x[shuffled_indices[batch_idx*batch_size: (batch_idx+1)*batch_size]]
            labels = y[shuffled_indices[batch_idx*batch_size: (batch_idx+1)*batch_size]]
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            net_acc += torch.sum((torch.max(outputs,1)[1] == labels).float()).data.item()
            ema_outputs = ema_net(inputs)
            ema_net_acc += torch.sum((torch.max(ema_outputs,1)[1] == labels).float()).data.item()
            global_step += 1
            update_ema_params(net, ema_net, global_step)
        if epoch % 50 == 0:
            print(f'Epoch: {epoch + 1}, Loss: {running_loss/num_batches :.3f}')
        net_acc /= len(x)
        ema_net_acc /= len(x)
        if net_acc - ema_net_acc < acc_diff:
            epoch_count += 1
            if epoch_count == 5:
                update_criteria = True
        else:
            epoch_count = 0
        update_cond = ((update_criteria == True or epoch == epochs-1) and scores_updated == False)
        if update_cond == True:
            print("Updating")
            hidden_embeddings = get_hidden_embeddings_diff(net, ema_net, x_unlabeled)
            scores = get_pseudo_loss_diff(net, ema_net, x_unlabeled)
            scores_updated = True
        acc_diff = net_acc - ema_net_acc
    return hidden_embeddings, scores, global_step

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
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]

        if sum(D2) == 0.0:
            pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll:
            ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll

def ablation(embeddings, hidden_diff, loss_diff, idxs_unlabeled, modified_indicies):
    hdiff_mod = 0.
    ldiff_mod = 0.
    hdiff_cle_ez = 0.
    ldiff_cle_ez = 0.
    embdg_mod = 0.
    embdg_cle_ez = 0.
    mod_cnt = 0
    cle_cnt_ez = 0

    for i in range(len(embeddings)):
        hdiff = np.linalg.norm(hidden_diff[i])
        ldiff = loss_diff[i][0]
        embdg = np.linalg.norm(embeddings[i])
        if idxs_unlabeled[i] in modified_indicies:
            hdiff_mod += hdiff
            ldiff_mod += ldiff
            embdg_mod += embdg
            mod_cnt += 1
        else:
            hdiff_cle_ez += hdiff
            ldiff_cle_ez += ldiff
            embdg_cle_ez += embdg
            cle_cnt_ez += 1

    hdiff_mod /= mod_cnt
    ldiff_mod /= mod_cnt
    hdiff_cle_ez /= cle_cnt_ez
    ldiff_cle_ez /= cle_cnt_ez
    embdg_mod /= mod_cnt
    embdg_cle_ez /= cle_cnt_ez
    print('  ** Modified examples:    hdiff={:.3f} ldiff={:.3f} embdg={:.3f} **'.format(hdiff_mod, ldiff_mod, embdg_mod))
    print('  ** Clean easy examples:  hdiff={:.3f} ldiff={:.3f} embdg={:.3f} **'.format(hdiff_cle_ez, ldiff_cle_ez, embdg_cle_ez))
    print('  ** Debug:                mod_cnt={} cle_cnt={}**'.format(mod_cnt, cle_cnt_ez))

def lhdiff_query(num_label_per_iteration, noisy_idxs, idxs_lb, data_size, hidden_embeddings, scores):
    idxs_unlabeled = np.arange(data_size)[~idxs_lb]
    hidden_diff = hidden_embeddings
    loss_diff = scores
    embeddings = hidden_diff * loss_diff.reshape(-1, 1)
    ablation(embeddings, hidden_diff, loss_diff.reshape(-1, 1), idxs_unlabeled, noisy_idxs)

    chosen = init_centers(embeddings, num_label_per_iteration)

    m_cnt, t_cnt = 0, 0
    for ix in range(len(chosen)):
        if idxs_unlabeled[chosen[ix]] in noisy_idxs:
            m_cnt += 1
        t_cnt += 1
    print('  ** {}/{} clean examples **'.format(t_cnt - m_cnt, t_cnt))
    return idxs_unlabeled[chosen]

def calc_fraction_dirty(to_label, noisy_idxs):
    num_noisy_selected = np.intersect1d(to_label, noisy_idxs)
    return len(num_noisy_selected) / len(to_label)

def visualize_decision_boundary(net, x, y):
    xx, yy = np.meshgrid(np.arange(-2, 7, 0.02),
                         np.arange(-1, 2, 0.02))
    predictions = net(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float))
    predictions = np.array(torch.argmax(predictions, dim=1))
    predictions = predictions.reshape(xx.shape)
    plt.clf()
    plt.contourf(xx, yy, predictions, alpha=.5)
    plt.scatter(x=x[:, 0], y=x[:, 1], c=y)
    plt.title(f"Decision boundary", fontsize=18)
    plt.xlim(-2., 7.)
    plt.ylim(-1., 1.5)
    plt.savefig(f"figures/Decision_boundary_data_points_{method}.png")
    plt.show()

if __name__ == "__main__":
    method = "lhdiff" # confidence or diversity or lhdiff
    verbose = True
    num_class = 4
    input_dimension = 2
    data_size = 30000
    batch_size = 16
    epochs = 100
    num_init_labelled = 20
    active_learning_iterations = 7
    num_label_per_iteration = 200
    net = None
    global_step = 0
    # generate initial labeled pool
    idxs_lb = torch.zeros(data_size, dtype=torch.bool)
    idxs_tmp = np.arange(data_size)
    np.random.shuffle(idxs_tmp)
    idxs_lb[idxs_tmp[:num_init_labelled]] = True
    x, y, test_x, test_y, noisy_idxs = create_moon_data(data_size=data_size, num_class=num_class, verbose=verbose)
    sampled_points = []
    plt.clf()
    plt.scatter(x=x[idxs_lb][:, 0], y=x[idxs_lb][:, 1], c=y[idxs_lb])
    plt.title("Initial labeled data")
    plt.xlim(-2., 7.)
    plt.ylim(-1., 1.5)
    plt.savefig("figures/Initial_labeled_data.png")
    plt.show()

    for i in range(active_learning_iterations):
        print('\nRound {}'.format(i+1))
        net = mlpMod(in_dim=input_dimension, out_dim=num_class, embSize=128)
        ema_net = mlpMod(in_dim=input_dimension, out_dim=num_class, embSize=128)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        hidden_embeddings, scores, global_step = train(net, ema_net, x[idxs_lb], y[idxs_lb], batch_size, epochs, optimizer, x[~idxs_lb], global_step)
        if method == "confidence":
            to_label = confidence_query(x, data_size, idxs_lb, num_label_per_iteration)
        elif method == "diversity":
            to_label = diversity_query(x, data_size, idxs_lb, num_label_per_iteration, net)
        else:
            to_label = lhdiff_query(num_label_per_iteration, noisy_idxs, idxs_lb, data_size, hidden_embeddings, scores)
        idxs_lb[to_label] = True
        sampled_points.extend(to_label)
    
    if verbose:
        print(f"Fraction Dirty: {calc_fraction_dirty(np.arange(data_size)[idxs_lb], noisy_idxs)}")
        visualize_decision_boundary(net, x, y)
        plt.clf()
        plt.scatter(x=x[idxs_lb][:, 0], y=x[idxs_lb][:, 1], c=y[idxs_lb])
        plt.title(f"Sampled data points", fontsize=18)
        plt.xlim(-2., 7.)
        plt.ylim(-1., 1.5)
        plt.savefig(f"figures/Sampled_data_{method}.png")
        plt.show()

        outputs = net(test_x)
        test_acc = torch.sum((torch.max(outputs, 1)[1] == test_y).float()).data.item() / len(test_y)
        print(f"Test Accuracy: {test_acc}")