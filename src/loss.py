
import torch
import torch.nn as nn
import src.utils
import torch.nn.functional as F
import numpy as np
import scipy.linalg as la

        



class SMPC_loss():
    
    def __init__(self, T=None, lam=0.1, rho=0.1, omega=None):
        
        self.T = T # number of samples
        self.lam = lam
        self.rho = rho
        self.omega=omega
        
    def compute_ker_quad_(self, Z, A=None): 
                
        if A is None:
            A = np.eye(Z.shape[0])
        return Z.T @ A @ Z
    
    
    def const_reg_(self, Z, Zt, Theta, rho=None):
        if rho is not None:
            self.rho=rho
        return (self.rho/2) * np.linalg.norm(Z - Theta @ Zt, ord='fro')**2
        
        
    
    
    def similarity_loss(self, X, Z):
        sim_term = (-1)/(2*self.T**2) * np.trace(self.compute_ker_quad_(X) @ self.compute_ker_quad_(Z)) + (1/(4*self.T**2)) * np.trace(self.compute_ker_quad_(Z)@self.compute_ker_quad_(Z))
        return sim_term
    
    
    
    def similarity_loss_with_L1(self, X, Z):
        
        sim_term = self.similarity_loss(X, Z)
        reg_term = (self.lam/self.T) * np.linalg.norm(Z, ord=1)
        return sim_term + reg_term, sim_term, reg_term
    
    
    def similarity_loss_with_L1_DL(self, X, Z, Zt, Theta):
        
        sim_term = self.similarity_loss(X, Zt)
        reg_term = (self.lam/self.T) * np.linalg.norm(Z, ord=1)
        dl_term = (1/self.T) * self.const_reg_(Z, Zt, Theta)
        
        return sim_term + reg_term + dl_term, sim_term, reg_term, dl_term
        
        
    
    
    def L1_loss(self, X, Z, H, lam=None):
        
        if lam is not None:
            self.lam = lam
        sim_term = (-1)/(2*self.T**2) * np.trace(self.compute_ker_quad_(X) @ self.compute_ker_quad_(Z, np.linalg.inv(H)))
        reg_term = (self.lam/self.T) * np.linalg.norm(Z, ord=1)
        
        return sim_term + reg_term, sim_term, reg_term
        
        
        
    def L1_loss_sim(self, X, Z, Zt, W):
        
        sim_term = ((-1)/self.T) * np.trace(X.T @ W.T @ Zt) + 0.5 * np.trace(W.T @ W)
        reg_term = (self.lam/self.T) * np.linalg.norm(Z, ord=1)
        
        return sim_term + reg_term, sim_term, reg_term
        
        
        
    def L1_loss_sim_with_constraint(self, X, Z, Zt, W, Theta, lam=0.1, rho=0.1):
        
        _, sim_term, reg_term = self.L1_loss_sim(X, Z, Zt, W, lam=lam)
        constraint_term = self.const_reg_(Z, X, rho=rho)
        
      
      
      

      
      
      
class SimilarityLossDL():
    
    def __init__(self, T=None, lam=0.1, rho=0.1, omega=None):
        
        self.T = T
        self.lam = lam
        self.rho = rho
        self.omega=omega
        
    def compute_ker_quad_(self, X, Z=None, A=None):
        
        if Z is None:
            Z = X.copy() 
                
        if A is None:
            A = np.eye(Z.shape[0])
        return Z.T @ A.T @ Z
    
    def sim_loss(self, X, Z):
        sim_term = (-1)/(2*self.T**2) * np.trace(self.compute_ker_quad_(X) @ self.compute_ker_quad_(Z)) + (1/(4*self.T**2)) * np.trace(self.compute_ker_quad_(Z)@self.compute_ker_quad_(Z))
        
        return sim_term
    
    
    def sim_loss_with_L1(self, X, Z):
        sim_term = self.sim_loss(X, Z)
        reg_term = (self.lam/self.T) * np.linalg.norm(Z, ord=1)
        return sim_term + reg_term, sim_term, reg_term
    
    
    def sim_loss_L1_DL(self, X, Z, Zt, Theta, lag=False, **kwargs):
        
        if not lag:
            sim_term = self.sim_loss(X, Zt)
            reg_term = (self.lam/self.T) * np.linalg.norm(Z, ord=1)
            const_reg = (self.rho/(2*self.T)) * np.linalg.norm(Z - Theta @ Zt, ord='fro')**2
            
        else:
            W = kwargs['W']
            M = kwargs['M']
            
            sim_term = (-1/self.T) * np.trace(self.compute_ker_quad_(X, Zt, W)) + (1/2)*np.trace(W.T @ W) + (1/(2*self.T)) * np.trace(self.compute_ker_quad_(Zt, A=M.T)) - (1/4) * np.trace(M.T @ M)
            reg_term = (self.lam/self.T) * np.linalg.norm(Z, ord=1)
            const_reg = (self.rho/(2*self.T)) * np.linalg.norm(Z - Theta @ Zt, ord='fro')**2
        
        return sim_term + reg_term + const_reg, sim_term, reg_term, const_reg
    
    
    
    def sim_loss_L1_DL_ver2(self, X, Z, Zt, Omega, lag=False, **kwargs):
        
        if not lag:
            sim_term = self.sim_loss(X, Zt)
            reg_term = (self.lam/self.T) * np.linalg.norm(Z, ord=1)
            const_reg = (self.rho/(2*self.T)) * np.linalg.norm(Zt - Omega @ Z, ord='fro')**2
            
        else:
            W = kwargs['W']
            M = kwargs['M']
            
            sim_term = (-1/self.T) * np.trace(self.compute_ker_quad_(X, Zt, W)) + (1/2)*np.trace(W.T @ W) + (1/(2*self.T)) * np.trace(self.compute_ker_quad_(Zt, A=M.T)) - (1/4) * np.trace(M.T @ M)
            reg_term = (self.lam/self.T) * np.linalg.norm(Z, ord=1)
            const_reg = (self.rho/(2*self.T)) * np.linalg.norm(Zt - Omega @ Z, ord='fro')**2
        
        return sim_term + reg_term + const_reg, sim_term, reg_term, const_reg
    
    
    def sim_loss_manifold_DL_ver2(self, X, Z, Zt, Omega, lag=False, **kwargs):
        
        if not lag:
            sim_term = self.sim_loss(X, Zt)
            # reg_term = (self.lam/self.T) * np.linalg.norm(Z, ord=1)
            const_reg = (self.rho/(2*self.T)) * np.linalg.norm(Zt - Omega @ Z, ord='fro')**2
            
        else:
            W = kwargs['W']
            M = kwargs['M']
            
            sim_term = (-1/self.T) * np.trace(self.compute_ker_quad_(X, Zt, W)) + (1/2)*np.trace(W.T @ W) + (1/(2*self.T)) * np.trace(self.compute_ker_quad_(Zt, A=M.T)) - (1/4) * np.trace(M.T @ M)
            # reg_term = (self.lam/self.T) * np.linalg.norm(Z, ord=1)
            const_reg = (self.rho/(2*self.T)) * np.linalg.norm(Zt - Omega @ Z, ord='fro')**2
        
        return sim_term + const_reg, sim_term, const_reg
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
    
        
        
        
    

            
            
        
            
        
        
        
        
        
        
        
        
        