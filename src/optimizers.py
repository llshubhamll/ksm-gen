import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from src.utils import sparsity_measure
from sparsemax import Sparsemax
# from src.run_utils import compute_rsquared, compute_similarity
from sys import exit
import matplotlib.pyplot as plt

class optimizationADMM:
    
    """
    KSM algorithm with fixed point update for the kernel matrix
    """
    def __init__(self, exp_params, model) -> None:
        super(optimizationADMM, self).__init__()
        self.K = model.K
        self.N = model.N
        self.lrs = exp_params['optimizer']['lrs']
        self.rho = model.rho
        self.latent_epochs = exp_params['optimizer']['latent_iters']
        self.param_epochs = exp_params['optimizer']['param_iters']
        self.model = model
        self.lam = exp_params['model']['lam']
        
    
    def set_values(self, input_data, variables, lags):
        
        self.input_data = input_data
        self.variables = variables
        self.lags = lags
        self.T = input_data.shape[0]
        # print(self.variables['Z'])
        
        
    def optim_variable(self, var_name) -> None:
        
        # self.model.param_no_grad()
        assert(self.model.W.requires_grad==False)
        if var_name not in self.lrs.keys():
            lr = self.lrs['Z']
        else:
            lr = self.lrs[var_name]
            
        if var_name == 'Z':
            for itercount in range(self.latent_epochs):
                # Doing sub gradient descent
                grad_values = self.compute_var_gradients(var_name)
                self.variables[var_name] = self.variables[var_name] - lr * grad_values
                
                # Learning rate decay
                if itercount % self.lrs['interval'] == 0:
                    lr = lr * self.lrs['Z_decay']
        
        elif var_name == 'P':
            constraint = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M'])
            W_corr = self.model.W @ self.model.W.T
            self.variables[var_name] = self.variables[var_name] + constraint - (W_corr /(2*self.rho))
            
                
    def optim_param(self, fixed=False) -> None:
        # self.model.param_set_grad()
        
        if fixed:
            H = torch.mean(self.variables['P'], dim=0)
            Z = self.variables['Z']
            X = self.input_data
            self.model.W = torch.linalg.solve(H, Z.T) @ X / (X.shape[0])
            
        else:
            for _ in range(self.param_epochs):
                grad_values = self.compute_param_gradients('W')
                self.model.W = self.model.W - self.lrs['params'] * grad_values
            
            
    def update_lags(self, lag_name) -> None:
        """
        Update lagrange multipliers
        """
        const = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M'])  

        self.lags[lag_name] = (self.rho) * const
        
                
                
    def get_loss(self) -> None:
        with torch.no_grad():
            loss, sim_term, l1_loss, aug_lag, sparsity = self.model(self.input_data, self.variables['Z'], self.variables['P'], self.lags['M'])
        
        return loss, sim_term, l1_loss, aug_lag, sparsity
    
    
    def compute_var_gradients(self, var_name):
        
        if var_name == 'Z':
            return self._calc_Z_grad()
        elif var_name == 'P':
            return self._calc_P_grad()
        
    
    def compute_param_gradients(self, param_name):
        
        if param_name == 'W':
            return self._calc_W_grad()
        
        else:
            raise NotImplementedError
        
        
    def _calc_W_grad(self):
        """
        Compute gradients with respect to W
        """
        act_term = - torch.mean(self.variables['Z'].unsqueeze(-1) @ self.input_data.unsqueeze(1), dim=0)
        H_term = torch.mean(self.variables['P'], dim=0) @ self.model.W
        grad_value = act_term + H_term
        return grad_value        
        
        
    def _calc_Z_grad(self):
        
        """
        Calculate the theoretical values based on formulation 1(c)
        """
        
        with torch.no_grad():
            const = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M'])
            grad_term1 = - (self.input_data/ self.T) @ self.model.W.T
            grad_term2 =  (2*self.model.rho/self.T) * const @ self.variables['Z'].unsqueeze(-1)
            grad_term3 =  self.model.lam / self.T * torch.sign(self.variables['Z'])
            
        return grad_term1 + grad_term2.squeeze() + grad_term3
    
    
    def _calc_P_grad(self):
        
        """
        Calculate the theoretical values of P gradients based on formulation 1(c)
        """
        with torch.no_grad():
            W_term = (0.5/self.T) * self.model.W @ self.model.W.T
            W_term = W_term.unsqueeze(0)
            const = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M'])
        
            aug_grad = - (self.model.rho / self.T) * const
            
        return W_term + aug_grad
    
    
    def _calc_P_optimal(self):
        with torch.no_grad():
            W_term = (0.5 /self.model.rho) * self.model.W @ self.model.W.T
            W_term = W_term.unsqueeze(0)
            const = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M']) + self.variables['P']
            
            return const - W_term





class optimizationADMM_proximal:
    
    """
    KSM algorithm with fixed point update for the kernel matrix and for Schedules 1 and 2.
    """
    def __init__(self, exp_params, model, eta=0.01) -> None:
        super(optimizationADMM_proximal, self).__init__()
        self.K = model.K
        self.N = model.N
        self.lrs = exp_params['optimizer']['lrs']
        self.rho = model.rho
        self.latent_epochs = exp_params['optimizer']['latent_iters']
        self.param_epochs = exp_params['optimizer']['param_iters']
        self.model = model
        self.eta = eta
        self.lam = exp_params['model']['lam']
        self.omega = model.omega
        self.device = exp_params['device']
        
    
    
    def update_eta(self, new_eta):
    
        self.eta = new_eta
    
    def set_values(self, input_data, variables, lags):
        
        self.input_data = input_data
        self.variables = variables
        self.lags = lags
        self.T = input_data.shape[0]
        # print(self.variables['Z'])
        
        
    def optim_variable(self, var_name) -> None:
        
        # print(f"Optimizing {var_name}")
        # self.model.param_no_grad()
        assert(self.model.W.requires_grad==False)
        if var_name not in self.lrs.keys():
            lr = self.lrs['Z']
        else:
            lr = self.lrs[var_name]
        
        update_norm = [] 
        if var_name == 'Z':
            W_corr = self.model.W @ self.model.W.T
            W_corr = W_corr.repeat(self.T, 1, 1)

            for itercount in range(self.latent_epochs):
                # Doing sub gradient descent
                Z_p = self.variables[var_name]
                grad_values = self.compute_var_gradients(var_name)
                # if (grad_values.isinf().any() == True):
                #     print('Inf detected')
                #     print(grad_values)
                #     print(itercount)
                #     print(self.variables['Z'])
                #     # exit(-1)
                Z_n = F.softshrink(Z_p - self.eta * grad_values, self.eta * self.lam / self.T)
                
                self.variables[var_name] = Z_n
                
                # update_norm.append(torch.norm(Z_n[0] - Z_p[0]))
                
                # Z_term = self.variables[var_name].unsqueeze(-1)
                # print("Z", self.variables['Z'][0])
                # print("Z_t", Z_term[0])
                
                # P =  Z_term @ Z_term.transpose(-1, -2) + self.omega * torch.eye(self.K, device=self.device)
                # self.variables['P'] = P
                constraint = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M'])
                self.variables['P'] = self.variables['P'] + constraint - (W_corr /(2*self.rho))
                
                # print("P", self.variables['P'][0])
                # print("P_nv", P[0])
                # print("M", self.lags['M'][0])
                # print("W_corr", W_corr[0]/2)
                
                
                # assert(torch.allclose(P, self.variables['P'], atol=1e-6) == True)


                
               
                
                
                # Learning rate decay
                if itercount % self.lrs['interval'] == 0:
                    lr = lr * self.lrs['Z_decay']
                    
                    
            # fig, ax = plt.subplots()
            # ax.plot(update_norm)
            # ax.set_xlabel('Proximal Iterations')
            # ax.set_ylabel('Update Norm')
            # fig.show()
        
        elif var_name == 'P':
            constraint = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M'])
            W_corr = self.model.W @ self.model.W.T
            self.variables[var_name] = self.variables[var_name] + constraint - (W_corr /(2*self.rho))
            
                
    def optim_param(self, fixed=False, norm=False) -> None:
        # self.model.param_set_grad()
        
        if fixed:
            H = torch.mean(self.variables['P'], dim=0)
            Z = self.variables['Z']
            X = self.input_data
            self.model.W = torch.linalg.solve(H, Z.T) @ X / (X.shape[0])
            
        else:
            for _ in range(self.param_epochs):
                grad_values = self.compute_param_gradients('W')
                self.model.W = self.model.W - self.lrs['params'] * grad_values
            
            # print(self.model.W[0])
    
            if norm:
                self.model.W /= torch.norm(self.model.W, dim=1, keepdim=True)
            
            
    def update_lags(self, lag_name, fixed=False) -> None:
        """
        Update lagrange multipliers
        """
        # print(f"Updating {lag_name}")
        const = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M'])  

        if fixed:
            print("Fixing Lagrange Multipliers")
            W_corr = self.model.W @ self.model.W.T / 2
            W_corr = W_corr.repeat(self.T, 1, 1)
            self.lags[lag_name] = W_corr
            
        else:
            self.lags[lag_name] = (self.rho) * const
        
        
                
                
    def get_loss(self) -> None:
        with torch.no_grad():
            loss, sim_term, l1_loss, aug_lag, sparsity = self.model(self.input_data, self.variables['Z'], self.variables['P'], self.lags['M'])
        
        return loss, sim_term, l1_loss, aug_lag, sparsity
    
    
    def compute_var_gradients(self, var_name):
        
        if var_name == 'Z':
            return self._calc_Z_smooth()
        elif var_name == 'P':
            return self._calc_P_grad()
        
    
    def compute_param_gradients(self, param_name):
        
        if param_name == 'W':
            return self._calc_W_grad()
        
        else:
            raise NotImplementedError
        
        
    def _calc_W_grad(self):
        """
        Compute gradients with respect to W
        """
        act_term = - torch.mean(self.variables['Z'].unsqueeze(-1) @ self.input_data.unsqueeze(1), dim=0)
        H_term = torch.mean(self.variables['P'], dim=0) @ self.model.W
        grad_value = act_term + H_term
        return grad_value        
        
        
    def _calc_Z_smooth(self):
        
        """
        Calculate the theoretical values based on formulation 1(c)
        """
        
        with torch.no_grad():
            const = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M'])
            grad_term1 = - (self.input_data / self.T) @ self.model.W.T
            grad_term2 =  (2*self.model.rho/self.T) * const @ self.variables['Z'].unsqueeze(-1)
        
        return grad_term1 + grad_term2.squeeze()
    
    
    def _calc_P_grad(self):
        
        """
        Calculate the theoretical values of P gradients based on formulation 1(c)
        """
        with torch.no_grad():
            W_term = (0.5/self.T) * self.model.W @ self.model.W.T
            W_term = W_term.unsqueeze(0)
            const = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M'])
        
            aug_grad = - (self.model.rho / self.T) * const
            
        return W_term + aug_grad
    
    
    def _calc_P_optimal(self):
        with torch.no_grad():
            W_term = (0.5 /self.model.rho) * self.model.W @ self.model.W.T
            W_term = W_term.unsqueeze(0)
            const = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M']) + self.variables['P']
            
            return const - W_term







class optimizationADMM_proximal_notrack:
    
    """
    KSM algorithm with fixed point update for the kernel matrix under Schedule 3.
    """
    def __init__(self, exp_params, model, eta=0.01) -> None:
        super(optimizationADMM_proximal_notrack, self).__init__()
        self.K = model.K
        self.N = model.N
        self.lrs = exp_params['optimizer']['lrs']
        self.rho = model.rho
        self.latent_epochs = exp_params['optimizer']['latent_iters']
        self.param_epochs = exp_params['optimizer']['param_iters']
        self.model = model
        self.eta = eta
        self.lam = exp_params['model']['lam']
        self.omega = model.omega
        self.device = exp_params['device']
        
    
    
    def update_eta(self, new_eta):
    
        self.eta = new_eta
    
    def set_values(self, input_data, variables, lags):
        
        self.input_data = input_data
        self.variables = variables
        self.lags = lags
        self.T = input_data.shape[0]
        # print(self.variables['Z'])
        
        
    def optim_variable(self, var_name) -> None:
        
        # print(f"Optimizing {var_name}")
        # self.model.param_no_grad()
        assert(self.model.W.requires_grad==False)
        if var_name not in self.lrs.keys():
            lr = self.lrs['Z']
        else:
            lr = self.lrs[var_name]
        
        update_norm = [] 
        if var_name == 'Z':
            

            for itercount in range(self.latent_epochs):
                # Doing sub gradient descent
                Z_p = self.variables[var_name]
                grad_values = self.compute_var_gradients(var_name)
                # if (grad_values.isinf().any() == True):
                #     print('Inf detected')
                #     print(grad_values)
                #     print(itercount)
                #     print(self.variables['Z'])
                #     # exit(-1)
                Z_n = F.softshrink(Z_p - self.eta * grad_values, self.eta * self.lam / self.T)
                
                self.variables[var_name] = Z_n
                
                update_norm.append(torch.norm(Z_n[0] - Z_p[0]))
                
                # Z_term = self.variables[var_name].unsqueeze(-1)
                # print("Z", self.variables['Z'][0])
                # print("Z_t", Z_term[0])
                
                # P =  Z_term @ Z_term.transpose(-1, -2) + self.omega * torch.eye(self.K, device=self.device)
                # self.variables['P'] = P
                # constraint = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M'])
                # self.variables['P'] = self.variables['P'] + constraint - (W_corr /(2*self.rho))
                
                # print("P", self.variables['P'][0])
                # print("P_nv", P[0])
                # print("M", self.lags['M'][0])
                # print("W_corr", W_corr[0]/2)
                
                
                # assert(torch.allclose(P, self.variables['P'], atol=1e-6) == True)


                
               
                
                
                # Learning rate decay
                if itercount % self.lrs['interval'] == 0:
                    lr = lr * self.lrs['Z_decay']
                    
                    
            # fig, ax = plt.subplots()
            # ax.plot(update_norm)
            # ax.set_xlabel('Proximal Iterations')
            # ax.set_ylabel('Update Norm')
            # fig.show()
        
        elif var_name == 'P':
            constraint = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M'])
            W_corr = self.model.W @ self.model.W.T
            W_corr = W_corr.repeat(self.T, 1, 1)
            self.variables[var_name] = self.variables[var_name] + constraint - (W_corr /(2*self.rho))
            
                
    def optim_param(self, fixed=False, norm=False) -> None:
        # self.model.param_set_grad()
        
        if fixed:
            H = torch.mean(self.variables['P'], dim=0)
            Z = self.variables['Z']
            X = self.input_data
            self.model.W = torch.linalg.solve(H, Z.T) @ X / (X.shape[0])
            
        else:
            for _ in range(self.param_epochs):
                grad_values = self.compute_param_gradients('W')
                self.model.W = self.model.W - self.lrs['params'] * grad_values
                
            # print(self.model.W[0])
            if norm:
                self.model.W /= torch.norm(self.model.W, dim=1, keepdim=True)
            
            
    def update_lags(self, lag_name, fixed=False) -> None:
        """
        Update lagrange multipliers
        """
        # print(f"Updating {lag_name}")
        const = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M'])  

        if fixed:
            print("Fixing Lagrange Multipliers")
            W_corr = self.model.W @ self.model.W.T / 2
            W_corr = W_corr.repeat(self.T, 1, 1)
            self.lags[lag_name] = W_corr
            
        else:
            self.lags[lag_name] = (self.rho) * const
        
        
                
                
    def get_loss(self) -> None:
        with torch.no_grad():
            loss, sim_term, l1_loss, aug_lag, sparsity = self.model(self.input_data, self.variables['Z'], self.variables['P'], self.lags['M'])
        
        return loss, sim_term, l1_loss, aug_lag, sparsity
    
    
    def compute_var_gradients(self, var_name):
        
        if var_name == 'Z':
            return self._calc_Z_smooth()
        elif var_name == 'P':
            return self._calc_P_grad()
        
    
    def compute_param_gradients(self, param_name):
        
        if param_name == 'W':
            return self._calc_W_grad()
        
        else:
            raise NotImplementedError
        
        
    def _calc_W_grad(self):
        """
        Compute gradients with respect to W
        """
        act_term = - torch.mean(self.variables['Z'].unsqueeze(-1) @ self.input_data.unsqueeze(1), dim=0)
        H_term = torch.mean(self.variables['P'], dim=0) @ self.model.W
        grad_value = act_term + H_term
        return grad_value        
        
        
    def _calc_Z_smooth(self):
        
        """
        Calculate the theoretical values based on formulation 1(c)
        """
        
        with torch.no_grad():
            const = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M'])
            grad_term1 = - (self.input_data / self.T) @ self.model.W.T
            grad_term2 =  (2*self.model.rho/self.T) * const @ self.variables['Z'].unsqueeze(-1)
        
        return grad_term1 + grad_term2.squeeze()
    
    
    def _calc_P_grad(self):
        
        """
        Calculate the theoretical values of P gradients based on formulation 1(c)
        """
        with torch.no_grad():
            W_term = (0.5/self.T) * self.model.W @ self.model.W.T
            W_term = W_term.unsqueeze(0)
            const = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M'])
        
            aug_grad = - (self.model.rho / self.T) * const
            
        return W_term + aug_grad
    
    
    def _calc_P_optimal(self):
        with torch.no_grad():
            W_term = (0.5 /self.model.rho) * self.model.W @ self.model.W.T
            W_term = W_term.unsqueeze(0)
            const = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M']) + self.variables['P']
            
            return const - W_term






class optimizationADMM_manifold:
    
    """
    KSM algorithm with fixed point update for the kernel matrix and the kernel structure being tracked
    """
    def __init__(self, exp_params, model, eta=0.01) -> None:
        super(optimizationADMM_manifold, self).__init__()
        self.K = model.K
        self.N = model.N
        self.lrs = exp_params['optimizer']['lrs']
        self.rho = model.rho
        self.latent_epochs = exp_params['optimizer']['latent_iters']
        self.param_epochs = exp_params['optimizer']['param_iters']
        self.model = model
        self.eta = eta
        self.omega = model.omega
        self.device = exp_params['device']
        self.sparsemax = Sparsemax(dim=-1)
        
        
    
    
    def update_eta(self, new_eta):
    
        self.eta = new_eta
    
    def set_values(self, input_data, variables, lags):
        
        self.input_data = input_data
        self.variables = variables
        self.lags = lags
        self.T = input_data.shape[0]
        # print(self.variables['Z'])
        
        
    def optim_variable(self, var_name) -> None:
        
        # print(f"Optimizing {var_name}")
        # self.model.param_no_grad()
        assert(self.model.W.requires_grad==False)
        if var_name not in self.lrs.keys():
            lr = self.lrs['Z']
        else:
            lr = self.lrs[var_name]
        
        update_norm = [] 
        if var_name == 'Z':
            W_corr = self.model.W @ self.model.W.T
            W_corr = W_corr.repeat(self.T, 1, 1)

            for itercount in range(self.latent_epochs):
                # Doing sub gradient descent
                Z_p = self.variables[var_name]
                grad_values = self.compute_var_gradients(var_name)
                Z_n = self.sparsemax(Z_p - self.eta * grad_values)
                
                self.variables[var_name] = Z_n
                
                # update_norm.append(torch.norm(Z_n[0] - Z_p[0]))
                
                # Z_term = self.variables[var_name].unsqueeze(-1)
                # print("Z", self.variables['Z'][0])
                # print("Z_t", Z_term[0])
                
                # P =  Z_term @ Z_term.transpose(-1, -2) + self.omega * torch.eye(self.K, device=self.device)
                # self.variables['P'] = P
                constraint = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M'])
                self.variables['P'] = self.variables['P'] + constraint - (W_corr /(2*self.rho))
                
                # print("P", self.variables['P'][0])
                # print("P_nv", P[0])
                # print("M", self.lags['M'][0])
                # print("W_corr", W_corr[0]/2)
                
                
                # assert(torch.allclose(P, self.variables['P'], atol=1e-6) == True)


                
               
                
                
                # Learning rate decay
                if itercount % self.lrs['interval'] == 0:
                    lr = lr * self.lrs['Z_decay']
                    
                    
            # fig, ax = plt.subplots()
            # ax.plot(update_norm)
            # ax.set_xlabel('Proximal Iterations')
            # ax.set_ylabel('Update Norm')
            # fig.show()
        
        elif var_name == 'P':
            constraint = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M'])
            W_corr = self.model.W @ self.model.W.T
            self.variables[var_name] = self.variables[var_name] + constraint - (W_corr /(2*self.rho))
            
                
    def optim_param(self, fixed=False, norm=False) -> None:
        # self.model.param_set_grad()
        
        if fixed:
            H = torch.mean(self.variables['P'], dim=0)
            Z = self.variables['Z']
            X = self.input_data
            self.model.W = torch.linalg.solve(H, Z.T) @ X / (X.shape[0])
            
        else:
            for _ in range(self.param_epochs):
                grad_values = self.compute_param_gradients('W')
                self.model.W = self.model.W - self.lrs['params'] * grad_values
            
            # print(self.model.W[0])
    
            if norm:
                self.model.W /= torch.norm(self.model.W, dim=1, keepdim=True)
            
            
    def update_lags(self, lag_name, fixed=False) -> None:
        """
        Update lagrange multipliers
        """
        # print(f"Updating {lag_name}")
        const = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M'])  

        if fixed:
            print("Fixing Lagrange Multipliers")
            W_corr = self.model.W @ self.model.W.T / 2
            W_corr = W_corr.repeat(self.T, 1, 1)
            self.lags[lag_name] = W_corr
            
        else:
            self.lags[lag_name] = (self.rho) * const
        
        
                
                
    def get_loss(self) -> None:
        with torch.no_grad():
            loss, sim_term, aug_lag = self.model(self.input_data, self.variables['Z'], self.variables['P'], self.lags['M'])
        
        return loss, sim_term, aug_lag
    
    
    def compute_var_gradients(self, var_name):
        
        if var_name == 'Z':
            return self._calc_Z_grad()
        elif var_name == 'P':
            return self._calc_P_grad()
        
    
    def compute_param_gradients(self, param_name):
        
        if param_name == 'W':
            return self._calc_W_grad()
        
        else:
            raise NotImplementedError
        
        
    def _calc_W_grad(self):
        """
        Compute gradients with respect to W
        """
        act_term = - (1+self.omega) * torch.mean(self.variables['Z'].unsqueeze(-1) @ self.input_data.unsqueeze(1), dim=0)
        H_term = torch.mean(self.variables['P'], dim=0) @ self.model.W
        grad_value = act_term + H_term
        return grad_value        
        
        
    def _calc_Z_grad(self):
        
        """
        Calculate the theoretical values based on formulation 1(c)
        """
        
        with torch.no_grad():
            const = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M'])
            grad_term1 = - (1+self.omega)*(self.input_data) @ self.model.W.T
            grad_term2 =  (2*self.model.rho) * const @ self.variables['Z'].unsqueeze(-1)
            # grad_term_diag = ((self.model.rho * self.omega) / self.T) * torch.diag_embed(torch.diagonal(const, dim1=1, dim2=2)) @ torch.ones(self.T, self.K, 1, device=self.device)
            grad_term_diag = ((self.model.rho * self.omega)) * const.diagonal(offset=0, dim1=1, dim2=2)
        
        return grad_term1 + grad_term2.squeeze() + grad_term_diag.squeeze()
    
    
    def _calc_P_grad(self):
        
        """
        Calculate the theoretical values of P gradients based on formulation 1(c)
        """
        with torch.no_grad():
            W_term = (0.5/self.T) * self.model.W @ self.model.W.T
            W_term = W_term.unsqueeze(0)
            const = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M'])
        
            aug_grad = - (self.model.rho / self.T) * const
            
        return W_term + aug_grad
    
    
    def _calc_P_optimal(self):
        with torch.no_grad():
            W_term = (0.5 /self.model.rho) * self.model.W @ self.model.W.T
            W_term = W_term.unsqueeze(0)
            const = self.model.const_val(self.variables['Z'], self.variables['P'], self.lags['M']) + self.variables['P']
            
            return const - W_term






