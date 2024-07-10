import matplotlib.pyplot as plt
import torch


def scatter_plot_2d(x, y, x_label=None, y_label=None, title=None, ax=None, **kwargs):
    """
    Scatter plot of 2D data.
    """
    if ax is None:
        fig, ax = plt.subplots()
    scatter = ax.scatter(x, y, **kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    return ax, scatter


def scatter_plot3d(x, y, z, x_label=None, y_label=None, z_label=None, title=None, ax=None, **kwargs):
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.add_axes(ax)
    scatter = ax.scatter(x, y, z, **kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    ax.view_init(azim=-60, elev=12)
    return ax, scatter
    




