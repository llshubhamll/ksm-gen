import torch
import torch.nn.functional as F
from sklearn import metrics


def compute_rsquared(D_W, D_est, variables, data):
    """
    Compute R^2 for a given model and data.
    data: (num_samples, N)
    variables: dictionary containing the following
        - P: (num_samples, K, K)
        - Z: (num_samples, K)
    """
    
    H_est = torch.mean(variables['P'], dim=0).cpu()
    Z_est = variables['Z'].cpu()
    
    # DW_est_n = D_W / torch.norm(D_W, dim=1, keepdim=True) 
    
    predictions = Z_est @ D_W
    r2_w = metrics.r2_score(data.numpy(), predictions.numpy())
    
    D_est = torch.linalg.solve(H_est, Z_est.T) @ data / (data.shape[0]) # (K, N)
    predictions = Z_est @ D_est
    
    r2_calc = metrics.r2_score(data.numpy(), predictions.numpy())
    
    return r2_w, r2_calc


def compute_similarity(tensor1, tensor2, dim=1):
    """
    Computes the mean cosine similarity between two tensors along a given dimension, with the mean computed along the first dimension.
    """
    sim_value = F.cosine_similarity(tensor1, tensor2, dim=dim).mean().item()
    
    return sim_value




        
        
    
class compute_run_metrics():
    
    """ 
    Compute different run metrics to track the optimization process
    Metrics include
    R2 score
    cosine similarity between true and estimated dictionary
    cosine similarity between true and estimated latent codes
    
    """
    def __init__(self, exp_params, data, true_dict=None, true_Z=None):
        self.exp_params = exp_params
        self.data = data
        self.true_dict = true_dict
        self.true_Z = true_Z
        
        
    def compute_r2(self, model, variables):
        """
        Function returns the R2 score for the model
        """
        # D_est = model.W.T.detach().cpu()
        H_est = torch.mean(variables['P'], dim=0)
        # print(H_est)
        Z_est = variables['Z']
        D_est = torch.linalg.solve(H_est, Z_est.T) @ self.data / self.data.shape[0]
        D_est_2 = model.W.detach()
        # D_est_2_n = D_est_2 / torch.linalg.norm(D_est_2, dim=1, keepdims=True)
        predictions = Z_est @ D_est
        r2 = metrics.r2_score(self.data.cpu().numpy(), predictions.cpu().numpy())
        predictions = Z_est @ D_est_2
        r2_2 = metrics.r2_score(self.data.cpu().numpy(), predictions.cpu().numpy())
        # print(r2)
        return r2, r2_2
    
    
    def compute_similarity(self, model, variables, dict_sim=True, latent_sim=True):
        """
        Function returns the arcsine of the cosine similarity between the estimated and true dictionary
        """
        if dict_sim:
            D_est = model.W.T.detach().cpu()
            D_est_n = D_est / torch.linalg.norm(D_est, dim=0, keepdims=True)
            D_true_n = self.true_dict / torch.linalg.norm(self.true_dict, dim=0, keepdims=True)
            d_sim = self._compute_cosine_similarity(D_est_n, D_true_n)
            
        else:
            d_sim = None
        
        # Compute similarity between latent codes
        if latent_sim:
            Z_est = variables['Z'].clone()
            Z_est = Z_est / torch.linalg.norm(Z_est, dim=1, keepdims=True)
            # print(Z_es/t)
            Z_true = self.true_Z.clone()
            Z_true = Z_true / torch.linalg.norm(Z_true, dim=1, keepdims=True)
            # print(Z_true)
            l_sim = self._compute_cosine_similarity(Z_est.T, Z_true.T)
            
        else:
            l_sim = None
        return d_sim, l_sim
    
    def _compute_cosine_similarity(self, tensor1, tensor2, dim=1):
        """
        Computes angle similarity between two tensors along feature dimension
        tensor1: n_features x n_components
        tensor2: n_features x n_components
        """
        cos_sim = F.cosine_similarity(tensor1, tensor2, dim=dim)
        return cos_sim.mean().item()
        # cos_sim = torch.sum(tensor1 * tensor2, dim=0)
        
        # print(cos_sim)
        # return (180/np.pi) * torch.mean(torch.arccos(torch.round(cos_sim, decimals=4)))
        # print('cos: ', cos_sim[cos_sim > 1])
        # sin_sim = torch.sqrt(1 - cos_sim**2)
        # print('sin', sin_sim)
        # return torch.mean(torch.arcsin(sin_sim)) * (180/np.pi)
    
    
    
    
    
    