import torch
import numpy as np
from sklearn.manifold import *
from sklearn.cluster import KMeans



def do_tsne(data, n_components=2, perplexity=30, n_iter=5000, seed=42):

    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=seed)
    tsne_results = tsne.fit_transform(data)
    
    return tsne_results, tsne


def do_spectral_embedding(data, n_components=2, affinity='nearest_neighbors', n_neighbors=10, n_jobs=1, seed=42):

    spectral_embedding = SpectralEmbedding(n_components=n_components, affinity=affinity, n_neighbors=n_neighbors, n_jobs=n_jobs, random_state=seed)
    spectral_embedding_results = spectral_embedding.fit_transform(data)
    
    return spectral_embedding_results, spectral_embedding



def do_kmeans(data, n_clusters=10, seed=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    kmeans.fit(data)
    return kmeans