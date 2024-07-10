import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils import soft_threshold, power_method_svd
import scipy.linalg as la
from scipy.linalg import sqrtm
import functorch
import matplotlib.pyplot as plt
from src.utils import sparsity_measure

       
            
class similarityMatching(nn.Module):
    
    """
    Traditional similarity matching module
    """
    
    
    def __init__(self, exp_params, seed=42, Z=None, W=None, M=None):
        super(similarityMatching, self).__init__()
        np.random.seed(seed)
        
        self.max_epochs = exp_params['optimizer']['max_epochs']
        self.lr_neuron = exp_params['optimizer']['lr_neuron']
        self.lr_param = exp_params['optimizer']['lr_param']
        self.K = exp_params['runtime']['K']
        self.N = exp_params['runtime']['N']
        self.T = exp_params['runtime']['T']
        if Z is None:
            self.Z = np.random.randn(self.K, self.T)
            
        else:
            self.Z = Z
            
        if W is None:
            self.W = np.random.randn(self.K, self.N)
            
        else:
            self.W = W
            
        if M is None:
            self.M = np.eye(self.K)
            
        else:
            self.M = M
        
    def update_neurons(self, X, iterate=True):
        
        if iterate:
            self.Z = self.Z + (self.lr_neuron/self.T) * (self.W @ X - self.M @ self.Z)
            
        else:
            self.Z = la.inv(self.M) @ self.W @ X
            
    
    def update_parameters(self, X, fixed=False):
        
        if not fixed:
            self.W = self.W + self.lr_param * ((self.Z @ X.T)/self.T - self.W)
            self.M = self.M + (self.lr_param/2) * ((self.Z @ self.Z.T)/self.T - self.M)
        
        else:
            self.W = (self.Z @ X.T) / self.T
            self.M = (self.Z @ self.Z.T) / self.T
        
        
    def return_neurons(self):
        return self.Z
    
    def return_parameters(self):
        return self.W, self.M
    
    
    
    
          


class SMPC(similarityMatching):
    
    def __init__(self, exp_params, seed=42):
        super().__init__(exp_params, seed)
        
        self.lam = exp_params['model']['lam']
        self.rho = exp_params['model']['rho']
        self.omega = exp_params['model']['omega']
        self.lr_omega = exp_params['optimizer']['lr_param_Omega']
        
        self.Z = np.random.laplace(size=(self.K, self.T))
        # self.latent = np.random.randn(self.K, self.T)
        self.latent = self.Z.copy()
        self.V = np.zeros((self.K, self.T))
        
        
        self.Hinv = self.compute_Hinv()
        self.H_inv_th = self.compute_Hinv()
        
        self.Omega = sqrtm(self.Hinv)
        
        self.error = self._compute_error(self.Omega, self.latent, self.Z, rho=self.rho)
        
    
        
        
        if 'step' in exp_params['model']:
            self.step = exp_params['model']['step']
            self.update_step = False
            
        else:
            self.step = self._compute_step_()
            self.update_step = True
        
        
        
        
    def _compute_step_(self, factor=0.5):
        return factor / power_method_svd(torch.from_numpy(self.Omega)).item()
        
    def _compute_error(self, Omega, Z, Zt, rho=1):
        return rho*(Zt - Omega @ Z)
    
    def compute_Hinv(self):
        return la.inv((1/self.T) * self.latent @ self.latent.T + self.omega*np.eye(self.K))
    
            
    def update_Z(self, X, fixed=False):
        
        # self.error = self.compute_error(self.Omega, self.latent, self.Z, rho=self.rho)
        
        self.Z = self.Z + (self.lr_neuron/self.T) * (self.W @ X - self.M @ self.Z - self.error)
        # self.Z = self.Z + (self.lr_neuron/self.T) * (self.W @ X - self.error)
        
        
    def update_latent(self):
        
        # self.V = self.V * (1-self.lr_neuron) + self.lr_neuron*((self.step / self.rho) * self.Omega.T @ self.error + self.latent)
        # self.error = self.compute_error(self.Omega, self.latent, self.Z, rho=self.rho)
        self.V = self.step  * self.Omega.T @ self.error + self.latent
        self.latent = soft_threshold(self.V, self.lam / (self.rho * self.step))
        
        
    def update_error(self, fixed=False):
        
        self.error = self.error + (self.lr_neuron/(self.rho*self.T)) * (self._compute_error(self.Omega, self.latent, self.Z, rho=self.rho) - self.error)
        
        
        
    def update_Omega(self, train=True, fixed=False, const=False):
        
        if train:
        
            if const:
                self.Omega = self.Omega + (self.lr_omega / self.T) * (self.error @ self.latent.T - self.Omega)
            
            else:
                self.Omega = self.Omega + (0.1*self.lr_omega / self.T) * (self.error @ self.latent.T) + (10*self.lr_omega / self.T) * (self.omega * np.eye(self.K) - (1/(2*self.T) * self.latent @ self.latent.T) -  self.Omega)
            self.Hinv = self.Omega.T @ self.Omega
            self.H_inv_th = self.compute_Hinv()
            
        else:
            
            self.Hinv = self.compute_Hinv()
            self.H_inv_th = self.Hinv.copy()
            self.Omega = sqrtm(self.Hinv)
            
        if self.update_step:
            self.step = self._compute_step_()
            
            
    def return_neurons(self):
        return {'Zt': self.Z, 'latent': self.latent, 'error': self.error}
    
    def return_parameters(self):
        return {'W': self.W, 'M': self.M, 'Omega': self.Omega}
        
        




 

class KSM_manifold(nn.Module):
    
    """
    Computes the KSM objective function
    """
    
    def __init__(self, exp_params, set_param=False, seed=42, W_init=None) -> None:
        super(KSM_manifold, self).__init__()
        torch.random.manual_seed(seed)
    
        # print(exp_params['model']['K'])
        self.K = exp_params['model']['K']
        self.N = exp_params['dataset']['dim']
        # self.lam = exp_params['model']['lam']
        self.omega = exp_params['model']['omega']
        self.rho = exp_params['model']['rho']
        self.device = exp_params['device']
        
        if W_init is None:
            self.W = torch.randn(self.K, self.N, dtype=torch.float32, requires_grad=False, device=self.device)
            self.W = self.W / torch.linalg.norm(self.W, dim=1, keepdim=True)
        else:
            self.W = W_init.clone().detach().to(self.device)
        if set_param:
            self.W = torch.nn.Parameter(self.W)
    
    
    def forward(self, X, Z, P, M):
        """
        Compute the objective.
        X: B X N
        Z: B X K
        P: B X K X K
        M: B X K X K
        W: K X N
        """        
        T = X.shape[0]
        # print(T)
        W_term = (-(1+self.omega)/T) * torch.trace(X @ self.W.T @ Z.T) + (0.5) * torch.trace(self.W.T @ torch.mean(P, dim=0) @ self.W) # torch.mean(functorch.vmap(torch.trace)(self.W.T @ P @ self.W))
        # sparsity = (self.lam/T) * torch.sum(torch.abs(Z))
        const_M = self.const_val(Z, P, M)
        constraint = const_M - M/self.rho
        penalty = 0.5 * self.rho * torch.mean(torch.linalg.norm(constraint, dim=(-1, -2), ord='fro')**2)
        lag = torch.trace(torch.mean(M.transpose(-1, -2) @ constraint, dim=0))
        aug_lag = lag + penalty
        loss_val = W_term  + aug_lag
        # sparse_measure = sparsity_measure(Z)
                
        return loss_val, W_term, penalty
    
    
    def const_val(self, Z, P, M):
        """
        Computes the term within the augmented lagrangian
        """
        if len(Z.shape) == 2:
            
            Z_term = Z.unsqueeze(-1)
        else:
            Z_term = Z
            
        T = Z_term.shape[0]
            
        const_M = Z_term @ Z_term.transpose(-1, -2) + self.omega * torch.diag_embed(Z_term.squeeze()) - P + M / (self.rho)
        
        return const_M
    
    
    # def _sparsity_measure(self, Z, p=1):
    #     """
    #     Computes the p-sparsity measure
    #     """
    #     m = torch.mean(Z, dim=1)
    #     n = Z.shape[1]
    #     Nr = Z - m.unsqueeze(-1)
    #     Nr_norm = torch.linalg.norm(Nr, dim=1, ord=p)
    #     Dr_norm = torch.linalg.norm(Z, dim=1, ord=p)
    #     C = (((n-1)**(p-1) + 1) / (n**(p-1)))**(1/p)
    #     sparsity_value = Nr_norm / (C * Dr_norm) * (n / (n-1))**(1/p)
    #     sparsity_value = torch.mean(sparsity_value.squeeze())
    #     return sparsity_value
        
    
    
    def param_no_grad(self):
        self.W.requires_grad = False
        
        return None
    
    def param_set_grad(self):
        self.W.requires_grad = True
        
        return None
    
 
 
 
 
 
 
 
 
 
class KSM_manifold_scaled(nn.Module):
    
    """
    Computes the KSM objective function
    """
    
    def __init__(self, exp_params, set_param=False, seed=42, W_init=None) -> None:
        super(KSM_manifold_scaled, self).__init__()
        torch.random.manual_seed(seed)
    
        # print(exp_params['model']['K'])
        self.K = exp_params['model']['K']
        self.N = exp_params['dataset']['dim']
        # self.lam = exp_params['model']['lam']
        self.omega = exp_params['model']['omega']
        self.rho = exp_params['model']['rho']
        self.device = exp_params['device']
        
        if W_init is None:
            self.W = torch.randn(self.K, self.N, dtype=torch.float32, requires_grad=False, device=self.device)
            self.W = self.W / torch.linalg.norm(self.W, dim=1, keepdim=True)
        else:
            self.W = W_init.clone().detach().to(self.device)
        if set_param:
            self.W = torch.nn.Parameter(self.W)
    
    
    def forward(self, X, Z, P, M):
        """
        Compute the objective.
        X: B X N
        Z: B X K
        P: B X K X K
        M: B X K X K
        W: K X N
        """        
        T = X.shape[0]
        # print(T)
        W_term = (-(1)/T) * torch.trace(X @ self.W.T @ Z.T) + (0.5) * torch.trace(self.W.T @ torch.mean(P, dim=0) @ self.W) # torch.mean(functorch.vmap(torch.trace)(self.W.T @ P @ self.W))
        # sparsity = (self.lam/T) * torch.sum(torch.abs(Z))
        const_M = self.const_val(Z, P, M)
        constraint = const_M - M/self.rho
        penalty = 0.5 * self.rho * torch.mean(torch.linalg.norm(constraint, dim=(-1, -2), ord='fro')**2)
        lag = torch.trace(torch.mean(M.transpose(-1, -2) @ constraint, dim=0))
        aug_lag = lag + penalty
        loss_val = W_term  + aug_lag
        # sparse_measure = sparsity_measure(Z)
                
        return loss_val, W_term, penalty
    
    
    def const_val(self, Z, P, M):
        """
        Computes the term within the augmented lagrangian
        """
        if len(Z.shape) == 2:
            
            Z_term = Z.unsqueeze(-1)
        else:
            Z_term = Z
            
        T = Z_term.shape[0]
            
        const_M = Z_term @ Z_term.transpose(-1, -2) + self.omega * torch.diag_embed(Z_term.squeeze()) - P + M / (self.rho)
        
        return const_M
    
    
    # def _sparsity_measure(self, Z, p=1):
    #     """
    #     Computes the p-sparsity measure
    #     """
    #     m = torch.mean(Z, dim=1)
    #     n = Z.shape[1]
    #     Nr = Z - m.unsqueeze(-1)
    #     Nr_norm = torch.linalg.norm(Nr, dim=1, ord=p)
    #     Dr_norm = torch.linalg.norm(Z, dim=1, ord=p)
    #     C = (((n-1)**(p-1) + 1) / (n**(p-1)))**(1/p)
    #     sparsity_value = Nr_norm / (C * Dr_norm) * (n / (n-1))**(1/p)
    #     sparsity_value = torch.mean(sparsity_value.squeeze())
    #     return sparsity_value
        
    
    
    def param_no_grad(self):
        self.W.requires_grad = False
        
        return None
    
    def param_set_grad(self):
        self.W.requires_grad = True
        
        return None
    
 
 
 
 
 
 

class KSM_objective(nn.Module):
    
    """
    Computes the KSM objective function
    """
    
    def __init__(self, exp_params, set_param=False, seed=42, W_init=None) -> None:
        super(KSM_objective, self).__init__()
        torch.random.manual_seed(seed)
    
        # print(exp_params['model']['K'])
        self.K = exp_params['model']['K']
        self.N = exp_params['dataset']['dim']
        self.lam = exp_params['model']['lam']
        self.omega = exp_params['model']['omega']
        self.rho = exp_params['model']['rho']
        self.device = exp_params['device']
        
        if W_init is None:
            self.W = torch.randn(self.K, self.N, dtype=torch.float32, requires_grad=False, device=self.device)
            self.W = self.W / torch.linalg.norm(self.W, dim=1, keepdim=True)
        else:
            self.W = W_init.clone().detach().to(self.device)
        if set_param:
            self.W = torch.nn.Parameter(self.W)
    
    
    def forward(self, X, Z, P, M):
        """
        Compute the objective.
        X: B X N
        Z: B X K
        P: B X K X K
        M: B X K X K
        W: K X N
        """        
        T = X.shape[0]
        # print(T)
        W_term = (-1/T) * torch.trace(X @ self.W.T @ Z.T) + (0.5) * torch.trace(self.W.T @ torch.mean(P, dim=0) @ self.W) # torch.mean(functorch.vmap(torch.trace)(self.W.T @ P @ self.W))
        sparsity = (self.lam/T) * torch.sum(torch.abs(Z))
        const_M = self.const_val(Z, P, M)
        constraint = const_M - M/self.rho
        penalty = 0.5 * self.rho * torch.mean(torch.linalg.norm(constraint, dim=(-1, -2), ord='fro')**2)
        lag = torch.trace(torch.mean(M.transpose(-1, -2) @ constraint, dim=0))
        aug_lag = lag + penalty
        loss_val = W_term + sparsity + aug_lag
        sparse_measure = sparsity_measure(Z)
                
        return loss_val, W_term, sparsity, penalty, sparse_measure
    
    
    def const_val(self, Z, P, M):
        """
        Computes the term within the augmented lagrangian
        """
        if len(Z.shape) == 2:
            
            Z_term = Z.unsqueeze(-1)
        else:
            Z_term = Z
            
        T = Z_term.shape[0]
            
        const_M = Z_term @ Z_term.transpose(-1, -2) + self.omega * torch.eye(self.K, device=self.device) - P + M / (self.rho)
        
        return const_M
    
    
    # def _sparsity_measure(self, Z, p=1):
    #     """
    #     Computes the p-sparsity measure
    #     """
    #     m = torch.mean(Z, dim=1)
    #     n = Z.shape[1]
    #     Nr = Z - m.unsqueeze(-1)
    #     Nr_norm = torch.linalg.norm(Nr, dim=1, ord=p)
    #     Dr_norm = torch.linalg.norm(Z, dim=1, ord=p)
    #     C = (((n-1)**(p-1) + 1) / (n**(p-1)))**(1/p)
    #     sparsity_value = Nr_norm / (C * Dr_norm) * (n / (n-1))**(1/p)
    #     sparsity_value = torch.mean(sparsity_value.squeeze())
    #     return sparsity_value
        
    
    
    def param_no_grad(self):
        self.W.requires_grad = False
        
        return None
    
    def param_set_grad(self):
        self.W.requires_grad = True
        
        return None
    
    