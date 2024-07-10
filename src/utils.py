
import random
from math import sqrt

import numpy as np

import torch.nn as nn
import torch, torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset


def power_method_svd(W, device, num_iters=200):
    """
    Returns SQUARE OF the largest singular value of W
    """
    x = torch.randn((W.shape[1], 1), dtype=torch.float32, device=device)    
            
    for i in range(num_iters):
        x = x / torch.norm(x)
        y = W @ x
        x = W.T @ y
        
    return torch.norm(x.reshape(-1))



def write_matrix_values(mat, ax):
    for (i, j), z in np.ndenumerate(mat):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=16)
 
 
 
 
def sparsity_measure(Z, p=1):
    """
    Computes the p-sparsity measure
    """
    m = torch.mean(Z, dim=1)
    n = Z.shape[1]
    Nr = Z - m.unsqueeze(-1)
    Nr_norm = torch.linalg.norm(Nr, dim=1, ord=p)
    Dr_norm = torch.linalg.norm(Z, dim=1, ord=p) + 1e-8
    C = (((n-1)**(p-1) + 1) / (n**(p-1)))**(1/p)
    sparsity_value = Nr_norm / (C * Dr_norm) * (n / (n-1))**(1/p)
    sparsity_value = torch.mean(sparsity_value.squeeze())
    return sparsity_value




def realign_dictionaries(cos_sim):
    """
    Function to realign dictionaries based on cosine similarity
    The dictionaries are initialized using Kmeans
    # true_D: (K, N)
    """
    K = cos_sim.shape[0]
    positions = np.ones(K, dtype=int) * (-1)
    indices = set(np.arange(K))
    for i in range(len(positions)):
        t = np.argmax(cos_sim[:, i])
        if cos_sim[t, i] == np.max(cos_sim[t, :]):
            positions[i] = t
            indices.remove(t)
            
    for i in range(len(positions)):
        if positions[i] == -1:
            pos = np.max(cos_sim[np.array(list(indices)), i])
            t = np.where(cos_sim[:, i] == pos)[0][0]
            positions[i] = t
            indices.remove(t)
            
    return positions
            
            
    
    