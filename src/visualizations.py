"""
Copyright (c) 2021 Bahareh Tolooshams

train for the model x = Dz

:author: Bahareh Tolooshams
"""

from locale import normalize
import numpy as np
import scipy as sp
import torch
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.gridspec as gridspec
import torchvision


def visualize_dense_dictionary(D, save_path=None, reshape=(28, 28), cmap="gray"):
    p = D.shape[-1]
    a = int(np.ceil(np.sqrt(p)))
    fig = plt.figure(figsize=(a, a))
    gs1 = gridspec.GridSpec(a, a)
    gs1.update(wspace=0.025, hspace=0.05)
    for col in range(p):
        ax1 = plt.subplot(gs1[col])
        wi = D[:, col].clone().detach().cpu().numpy()
        if reshape:
            wi = np.reshape(wi, reshape)
        plt.imshow(wi, cmap=cmap)
        plt.axis("off")
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect("equal")
        plt.subplots_adjust(wspace=None, hspace=None)
    # plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    # plt.close()
    return fig



def visualize_image_grid(img_tensor, nrow=None, ncol=None, normalize=False, ax=None):
    """
        Visualize Image tensor in a grid form
        
        Arguments
        =========

        img_tensor: Input tensor of dimension N X C X W X H

    """
    if ax is None:
        _, ax = plt.subplots()
    

    if nrow is None or ncol is None:
        nrow = int(np.sqrt(img_tensor.shape[0]))
        ncol = nrow
    # print(nrow, ncol)

    img_grid = torchvision.utils.make_grid(img_tensor, nrow=nrow, normalize=True)
    # print(img_grid.shape, img_tensor.shape)
    # print(torch.max(img_grid), torch.min(img_grid))
    img_grid = np.transpose(img_grid.detach().cpu().numpy(), (1, 2, 0))

    # fig.subplots_adjust(left=0, right=0.9, bottom=0, top=1)
    # cax = fig.add_axes([0.91, 0.05, 0.04, 0.9])

    # img_scale = (img_grid - img_grid.min(axis=(0, 1), keepdims=True)) / (img_grid.max(axis=(0, 1), keepdims=True) - img_grid.min(axis=(0, 1), keepdims=True))
    # print(np.max(img_scale), np.min(img_scale))
    if (img_tensor.shape[-1] == 1):
       im = ax.imshow(img_grid, cmap='gray')
       print("gray")
    else:
        im = ax.imshow(img_grid) 
        # print("rgb")
        # print(im)
        # fig.colorbar(im, cax=cax)
    ax.set_axis_off()
    return ax




    



def visualize_latents(Z_vals, num_samples=5, ax=None, seed=42, **kwargs):
    
    # Set figure width and height based on number of samples
    fig_width = 5
    fig_height = 3 * num_samples
    if ax is None:
        fig, ax = plt.subplots(num_samples, 1, figsize=(fig_width, fig_height))
        
    np.random.seed(seed)
    idx = np.random.choice(Z_vals.shape[0], num_samples, replace=False)
    
    for i, idx in enumerate(idx):
        ax[i].stem(np.abs(Z_vals[idx]), **kwargs)
        ax[i].set_title(f'Sample {idx}')
        # ax[i].grid(False)
        
    return ax
   