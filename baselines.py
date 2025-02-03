import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


"""

Unfortunately, I did not get this to work in time.

"""

# Import datasets from dgl
cora = dgl.data.CoraGraphDataset()
citeseer = dgl.data.CiteseerGraphDataset()
pubmed = dgl.data.PubmedGraphDataset()

def symmetrically_normalize(A):
    # To efficiently invert large D:
    D_inv_sqrt = torch.sum(A, dim=1)
    D_inv_sqrt = torch.sqrt(D_inv_sqrt)
    D_inv_sqrt = 1 / D_inv_sqrt
    D_inv_sqrt = torch.diag(D_inv_sqrt)

    # Efficient matrix multiplication of sparce matrices
    A_tilde = D_inv_sqrt.to_sparse_coo() @ A
    A_tilde = A_tilde.to_sparse_coo() @ D_inv_sqrt

    return A_tilde

def prepare_data(dataset, val_size=0.05, test_size=0.1):
    X = dataset[0].ndata['feat']
    A = dataset[0].adj_external().to_dense()
    
    assert (A == A.T).all()

    # Split the edges into train, val and test
    edges = torch.nonzero(A, as_tuple=False)

    # In symmetric A: [i, j] == [j, i]
    edges = torch.sort(edges, dim=1).values
    edges = torch.unique(edges, dim=0)

    num_edges = edges.shape[0]
    edges = edges[torch.randperm(num_edges)]

    num_val = int(val_size * num_edges)
    num_test = int(test_size * num_edges)

    # Save positive validation/test edges
    val_edges = edges[:num_val]
    test_edges = edges[num_val : num_val + num_test]
    
    # Mask A_train
    A_train = A.clone()
    A_train[val_edges[:, 0], val_edges[:, 1]] = 0
    A_train[val_edges[:, 1], val_edges[:, 0]] = 0
    A_train[test_edges[:, 1], test_edges[:, 0]] = 0
    A_train[test_edges[:, 0], test_edges[:, 1]] = 0

    # Add self-loops
    A = torch.eye(A.shape[0]) + A
    A_train = torch.eye(A_train.shape[0]) + A_train

    # Sample negative validation/test edges
    num_samples = num_test * 2 # oversample to ensure enough negative samples

    row_idx = torch.randint(0, A.shape[0], (num_samples,))
    col_idx = torch.randint(0, A.shape[0], (num_samples,))
    samples = torch.stack((row_idx, col_idx), dim=1)

    # In symmetric A: [i, j] == [j, i]
    samples = torch.sort(samples, dim=1).values
    samples = torch.unique(samples, dim=0)

    # Delete indices where A_ij = 1
    mask = A[samples[:, 0], samples[:, 1]] == 0
    samples = samples[mask]

    # Allocate samples
    try:
        val_non_edges = samples[:num_val]
        test_non_edges = samples[num_val:num_val + num_test]
    except IndexError:
        raise ValueError(
            "Not enough negative samples were found. Oversample more."
            )

    # Calculate A_tilde = D^-0.5 * A * D^-0.5
    A_tilde = symmetrically_normalize(A)
    A_tilde_train = symmetrically_normalize(A_train)

    I = torch.eye(A.shape[0])

    assert len(val_edges) == len(val_non_edges)
    assert len(test_edges) == len(test_non_edges)
    assert test_edges.shape[1] == 2
    assert torch.unique(test_edges, dim=0).shape[0] == test_edges.shape[0]
    assert torch.unique(test_non_edges, dim=0).shape[0] == test_non_edges.shape[0]
    assert A_tilde.shape == A_tilde_train.shape
    assert A_tilde.shape == A.shape
    assert A_tilde.shape == A_train.shape
    assert A_tilde.shape[0] == X.shape[0]

    return {'X': X, 'I': I, 'A': A, 'A_tilde': A_tilde,
            'masked': {'A': A_train, 'A_tilde': A_tilde_train},
            'val': {'edges': val_edges, 'non-edges': val_non_edges},
            'test': {'edges': test_edges, 'non-edges': test_non_edges}}

'''
Evaluation metrics
'''

def area_under_roc_curve(A_hat, edges, non_edges):
    '''
    edges / non-edges: either from val or test set
    '''
    thresholds = torch.linspace(0, 1, 100)
    TPRs = []
    FPRs = []

    assert torch.all((A_hat >= 0) & (A_hat <= 1))
    
    real = A_hat[edges[:, 0], edges[:, 1]]
    fake = A_hat[non_edges[:, 0], non_edges[:, 1]]
    
    for threshold in thresholds:
        TP = (real >= threshold).sum().item()
        FP = (fake >= threshold).sum().item()
        
        FN = (real < threshold).sum().item()
        TN = (fake < threshold).sum().item()
        
        TPR = TP / (TP + FN) if TP + FN != 0 else 0
        FPR = FP / (FP + TN) if FP + TN != 0 else 0
        
        TPRs.append(TPR)
        FPRs.append(FPR)
    
    TPRs = torch.tensor(TPRs)
    FPRs = torch.tensor(FPRs)

    sorted_indices = torch.argsort(FPRs)
    FPRs = FPRs[sorted_indices]
    TPRs = TPRs[sorted_indices]

    assert torch.all(TPRs >= 0) and torch.all(TPRs <= 1)
    assert torch.all(FPRs >= 0) and torch.all(FPRs <= 1)
    
    # Compute AUC using the trapezoidal rule
    auc = torch.trapz(TPRs, FPRs).item()

    assert auc >= 0 and auc <= 1

    # Check if the AUC is correct
    preds = A_hat[edges[:, 0], edges[:, 1]].detach().numpy()
    preds_neg = A_hat[non_edges[:, 0], non_edges[:, 1]].detach().numpy()
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    
    return roc_score

def average_precision(A_hat, edges, non_edges):

    real = A_hat[edges[:, 0], edges[:, 1]]
    fake = A_hat[non_edges[:, 0], non_edges[:, 1]]

    scores = torch.cat([real, fake])
    labels = torch.cat([torch.ones_like(real), torch.zeros_like(fake)])

    sorted_indices = torch.argsort(scores, descending=True)
    labels = labels[sorted_indices]

    tp = labels.cumsum(0)  # Cumulative sum of true positives
    fp = torch.arange(1, len(labels) + 1) - tp

    recall = tp / tp[-1]
    precision = tp / (tp + fp)

    ap = torch.sum(precision * torch.diff(torch.cat([torch.tensor([0.0]), recall])))

    # Check if the AP is correct
    preds = A_hat[edges[:, 0], edges[:, 1]].detach().numpy()
    preds_neg = A_hat[non_edges[:, 0], non_edges[:, 1]].detach().numpy()
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    ap_score = average_precision_score(labels_all, preds_all)

    return ap_score


# Save the datasets in a dictionary
data = {'cora': prepare_data(cora), 'citeseer': prepare_data(citeseer)} # 'pubmed': prepare_data(pubmed)
print('Done preparing data.')

'''
Baselines
'''

from sklearn.manifold import SpectralEmbedding

def spectral_clustering_baseline(data_dict, embedding_dim=128):
    A = data_dict['A'].clone()
    A_np = A.numpy()
    se = SpectralEmbedding(n_components=embedding_dim, affinity='precomputed')
    Z_np = se.fit_transform(A_np)
    Z = torch.tensor(Z_np, dtype=torch.float)
    return torch.sigmoid(Z @ Z.t())


'''
Main
'''

data = {'cora': prepare_data(cora), 'citeseer': prepare_data(citeseer)}


for dataset in ['cora', 'citeseer']:
    print(f"\nDataset: {dataset}")

    A_hat_sc = spectral_clustering_baseline(data[dataset])