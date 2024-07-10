import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from src.utils import sparsity_measure, power_method_svd
from src.run_utils import compute_rsquared, compute_similarity
from sys import exit





def ADMM_sparsecoder(model, optimizer, data, variables, lagrange,
             dataloader, exp_params, device, true_vals=None, 
             result_path=None, wandb_params=None):
    """
    This function implements and runs the ADMM algorithm for the given model and optimizer.
    """
    loss_vals = {'total_loss': [], 'similarity_loss': [], 'l1_loss': [], 'penalty': [], 'sparsity': [], 'r2_W': [], 'r2_est': [], 'dw_sim': [], 'dest_sim':[], 'latent_sim': []}
    
    pbar = tqdm(range(exp_params['optimizer']['max_epochs']), desc='Training', position=0, leave=True)
    
    T = variables['Z'].shape[0]
    with torch.no_grad():
        for epoch in pbar:
            
            est_loss = 0
            sim_loss = 0
            l1_loss = 0
            sparsity = 0
            penalty = 0
    
            for batch_idx, (input_batch, index) in enumerate(dataloader):
                
                # print(input_batch.to(device), index.shape)
                # return
                input_batch = input_batch.to(device)
                # Z = variables['Z'][index].clone()
                # P = variables['P'][index].clone()
                # M = lagrange['M'][index].clone()
                Z = variables['Z'][index]
                P = variables['P'][index]
                M = lagrange['M'][index]
                # M = model.W @ model.W.T / 2
                # M = M.repeat(input_batch.shape[0], 1, 1)
                batch_variables = {'Z': Z, 'P': P}
                batch_lagrange = {'M': M}
                
                optimizer.set_values(input_batch, batch_variables, batch_lagrange)
                
                optimizer.optim_variable('Z')
                # optimizer.optim_variable('P')
                optimizer.update_lags('M')
                
                if torch.isnan(optimizer.variables['Z']).any() or optimizer.variables['Z'].isinf().any():
                    if torch.isnan(optimizer.variables['Z']).any():
                        print('Nan detected')
                        print(batch_idx, epoch)
                        raise ValueError('Nan detected')
                    else:
                        print('Inf detected')
                        print(batch_idx, epoch)
                        raise ValueError('Inf detected')
                    
                    
                    
                # optimizer.optim_param()
                
                # variables['Z'][index, :] = optimizer.variables['Z'].clone()
                # variables['P'][index, :] = optimizer.variables['P'].clone()
                # lagrange['M'][index, :] = optimizer.lags['M'].clone()
                
                
                variables['Z'][index] = optimizer.variables['Z']
                variables['P'][index] = optimizer.variables['P']
                lagrange['M'][index] = optimizer.lags['M']
                
                # return
                
                if (epoch % exp_params['optimizer']['log_interval'] == 0) or (epoch == exp_params['optimizer']['max_epochs'] - 1):
                    
                    loss_b, sim_b, l1_loss_b, penalty_b, sparsity_b = optimizer.get_loss()
                    
                    est_loss += loss_b.item() * exp_params['dataset']['batch_size']
                    sim_loss += sim_b.item() * exp_params['dataset']['batch_size']
                    l1_loss += l1_loss_b.item() * exp_params['dataset']['batch_size']
                    penalty += penalty_b.item() * exp_params['dataset']['batch_size']
                    sparsity += sparsity_b.item() * exp_params['dataset']['batch_size']
                
                
                # return
            if (epoch % exp_params['optimizer']['log_interval'] == 0) or (epoch == exp_params['optimizer']['max_epochs'] - 1):
                est_loss /= T
                sim_loss /= T
                sparsity /= T
                penalty /= T
                l1_loss /= T
    

        
                D_W = model.W.cpu()
                H_est = torch.mean(variables['P'], dim=0).cpu()
                Z_est = variables['Z'].cpu()
                # assert(torch.linalg.det(H_est) != 0)
                D_est = torch.linalg.solve(H_est, Z_est.T) @ data / (data.shape[0]) # (K, N)
                r2_w, r2 = compute_rsquared(D_W, D_est, variables, data)

                if true_vals:
                    D_true = true_vals['D']

                    d_sim = compute_similarity(D_true, D_est)
                    dw_sim = compute_similarity(D_true, D_W)
                
                    Z_true = true_vals['Z']
                    z_sim = compute_similarity(Z_true, Z_est)
                    loss_vals['dw_sim'].append(dw_sim)
                    loss_vals['dest_sim'].append(d_sim)
                    loss_vals['latent_sim'].append(z_sim)
                    
                else:
                    dw_sim = 0
                    d_sim = 0
                    z_sim = 0


                
                loss_vals['total_loss'].append(est_loss)
                loss_vals['similarity_loss'].append(sim_loss)
                loss_vals['l1_loss'].append(l1_loss)
                loss_vals['penalty'].append(penalty)
                loss_vals['sparsity'].append(sparsity)
                loss_vals['r2_W'].append(r2_w)
                # print(r2)
                loss_vals['r2_est'].append(r2)            
                pbar.set_postfix_str(f"Total Loss: {est_loss:.4f} | Similarity Loss: {sim_loss:.4f} | L1 Loss: {sparsity:.4f} | Penalty: {penalty:.4f} | R2_W: {r2_w:.4f} | R2_est: {r2:.4f} | D_sim: {d_sim:.4f} | Z_sim: {z_sim:.4f} | DW_sim: {dw_sim:.4f} | sparsity: {sparsity:.4f}")
        
            # return  
                
    return loss_vals


# def run_ADMM_v2(model, optimizer, data, variables, lagrange,
#              dataloader, exp_params, true_vals, device,
#              result_path=None, wandb_params=None, update_eta=False):
#     """
#     This function implements and runs the ADMM algorithm for the given model and optimizer.
#     """
#     loss_vals = {'total_loss': [], 'similarity_loss': [], 'l1_loss': [], 'penalty': [], 'sparsity': [], 'r2_W': [], 'r2_est': [], 'dw_sim': [], 'dest_sim':[], 'latent_sim': []}
    
#     pbar = tqdm(range(exp_params['optimizer']['max_epochs']), desc='Training', position=0, leave=True)
    
#     new_eta = optimizer.eta
#     T = variables['Z'].shape[0]
#     with torch.no_grad():
#         for epoch in pbar:
            
#             est_loss = 0
#             sim_loss = 0
#             l1_loss = 0
#             sparsity = 0
#             penalty = 0
    
#             for batch_idx, (input_batch, index) in enumerate(dataloader):
                
#                 # print(input_batch.to(device), index.shape)
#                 # return
#                 input_batch = input_batch.to(device)
#                 Z = variables['Z'][index]
#                 P = variables['P'][index]
#                 M = lagrange['M'][index]
#                 batch_variables = {'Z': Z, 'P': P}
#                 batch_lagrange = {'M': M}
                
#                 optimizer.set_values(input_batch, batch_variables, batch_lagrange)
                
#                 if update_eta:
#                     new_eta = 1 / power_method_svd(model.W.T, exp_params['device'])
#                     # print(new_eta)
#                     optimizer.update_eta(new_eta)
                


#                 optimizer.optim_variable('Z')
#                 optimizer.optim_variable('P')
#                 optimizer.update_lags('M')
                
#                 if torch.isnan(optimizer.variables['Z']).any() or optimizer.variables['Z'].isinf().any():
#                     if torch.isnan(optimizer.variables['Z']).any():
#                         print('Nan detected')
#                         print(batch_idx, epoch)
#                         raise ValueError('Nan detected')
#                     else:
#                         print('Inf detected')
#                         print(batch_idx, epoch)
#                         raise ValueError('Inf detected')
                    
#                 # print(optimizer.variables['Z'])
#                 # print(optimizer.variables['P'])
#                 # print(model.W)
                    
                
#                 if torch.isnan(model.W).any() or model.W.isinf().any():
#                     if torch.isnan(model.W).any():
#                         print('param Nan detected')
#                         print(batch_idx, epoch)
#                         raise ValueError('param Nan detected')
#                     else:
#                         print('param Inf detected')
#                         print(batch_idx, epoch)
#                         raise ValueError('param Inf detected')
#                 # variables['Z'][index, :] = optimizer.variables['Z'].clone()
#                 # variables['P'][index, :] = optimizer.variables['P'].clone()
#                 # lagrange['M'][index, :] = optimizer.lags['M'].clone()
#                 variables['Z'][index] = optimizer.variables['Z']
#                 variables['P'][index] = optimizer.variables['P']
#                 lagrange['M'][index] = optimizer.lags['M']
                
                
                
                
                
                
#             # Update W
            
#             for batch_idx, (input_batch, index) in enumerate(dataloader):
#                 input_batch = input_batch.to(device)
#                 Z = variables['Z'][index]
#                 P = variables['P'][index]
#                 M = lagrange['M'][index]
#                 batch_variables = {'Z': Z, 'P': P}
#                 batch_lagrange = {'M': M}
                
#                 optimizer.set_values(input_batch, batch_variables, batch_lagrange)
#                 optimizer.optim_param()
#                 # return
                
#                 if epoch % exp_params['optimizer']['log_interval'] == 0:
                    
#                     loss_b, sim_b, l1_loss_b, penalty_b, sparsity_b = optimizer.get_loss()
                    
#                     est_loss += loss_b.item() * exp_params['dataset']['batch_size']
#                     sim_loss += sim_b.item() * exp_params['dataset']['batch_size']
#                     l1_loss += l1_loss_b.item() * exp_params['dataset']['batch_size']
#                     penalty += penalty_b.item() * exp_params['dataset']['batch_size']
#                     sparsity += sparsity_b.item() * exp_params['dataset']['batch_size']
                
                
#                 # return
#             if epoch % exp_params['optimizer']['log_interval'] == 0:
#                 est_loss /= T
#                 sim_loss /= T
#                 sparsity /= T
#                 penalty /= T
    

        
#                 D_true = true_vals['D']
#                 D_W = model.W.cpu()
#                 H_est = torch.mean(variables['P'], dim=0).cpu()
#                 Z_est = variables['Z'].cpu()
#                 D_est = torch.linalg.solve(H_est, Z_est.T) @ data / (data.shape[0]) # (K, N)
                
                    
#                 r2_w, r2 = compute_rsquared(D_W, D_est, variables, data)
#                 d_sim = compute_similarity(D_true, D_est)
#                 dw_sim = compute_similarity(D_true, D_W)
                
#                 Z_true = true_vals['Z']
#                 z_sim = compute_similarity(Z_true, Z_est)
                
#                 loss_vals['total_loss'].append(est_loss)
#                 loss_vals['similarity_loss'].append(sim_loss)
#                 loss_vals['l1_loss'].append(sparsity)
#                 loss_vals['penalty'].append(penalty)
#                 loss_vals['sparsity'].append(sparsity)
#                 loss_vals['r2_W'].append(r2_w)
#                 loss_vals['r2_est'].append(r2)
#                 loss_vals['dw_sim'].append(dw_sim)
#                 loss_vals['dest_sim'].append(d_sim)
#                 loss_vals['latent_sim'].append(z_sim)
                
                
#                 pbar.set_postfix_str(f"Total Loss: {est_loss:.4f} | Similarity Loss: {sim_loss:.4f} | L1 Loss: {sparsity:.4f} | Penalty: {penalty:.4f} | R2_W: {r2_w:.4f} | R2_est: {r2:.4f} | D_sim: {d_sim:.4f} | Z_sim: {z_sim:.4f} | DW_sim: {dw_sim:.4f} | new_eta: {new_eta:.4f}")
        
#             # return  
                
#         return loss_vals








# def ADMM_sparsecoder_altmin(model, optimizer, data, variables, lagrange,
#              dataloader, exp_params, true_vals, device,
#              result_path=None, wandb_params=None):
#     """
#     This function implements and runs the ADMM algorithm for the given model and optimizer.
#     """
#     loss_vals = {'total_loss': [], 'similarity_loss': [], 'l1_loss': [], 'penalty': [], 'sparsity': [], 'r2_W': [], 'r2_est': [], 'dw_sim': [], 'dest_sim':[], 'latent_sim': []}
    
#     pbar = tqdm(range(exp_params['optimizer']['max_epochs']), desc='Training', position=0, leave=True)
    
#     T = variables['Z'].shape[0]
#     with torch.no_grad():
#         for epoch in pbar:
            
#             est_loss = 0
#             sim_loss = 0
#             l1_loss = 0
#             sparsity = 0
#             penalty = 0
    
#             for batch_idx, (input_batch, index) in enumerate(dataloader):
                
#                 # print(input_batch.to(device), index.shape)
#                 # return
#                 input_batch = input_batch.to(device)
#                 Z = variables['Z'][index].clone()
#                 P = variables['P'][index].clone()
#                 M = lagrange['M'][index].clone()
#                 batch_variables = {'Z': Z, 'P': P}
#                 batch_lagrange = {'M': M}
                
#                 optimizer.set_values(input_batch, batch_variables, batch_lagrange)
                
#                 optimizer.optim_variable('Z')
#                 optimizer.optim_variable('P')
#                 optimizer.update_lags('M')
                
#                 if torch.isnan(optimizer.variables['Z']).any() or optimizer.variables['Z'].isinf().any():
#                     if torch.isnan(optimizer.variables['Z']).any():
#                         print('Nan detected')
#                         print(batch_idx, epoch)
#                         raise ValueError('Nan detected')
#                     else:
#                         print('Inf detected')
#                         print(batch_idx, epoch)
#                         raise ValueError('Inf detected')
                    
                    
                    
#                 # optimizer.optim_param()
                
#                 variables['Z'][index, :] = optimizer.variables['Z'].clone()
#                 variables['P'][index, :] = optimizer.variables['P'].clone()
#                 lagrange['M'][index, :] = optimizer.lags['M'].clone()
                
#                 # return
                
#                 if epoch % exp_params['optimizer']['log_interval'] == 0:
                    
#                     loss_b, sim_b, l1_loss_b, penalty_b, sparsity_b = optimizer.get_loss()
                    
#                     est_loss += loss_b.item() * exp_params['dataset']['batch_size']
#                     sim_loss += sim_b.item() * exp_params['dataset']['batch_size']
#                     l1_loss += l1_loss_b.item() * exp_params['dataset']['batch_size']
#                     penalty += penalty_b.item() * exp_params['dataset']['batch_size']
#                     sparsity += sparsity_b.item() * exp_params['dataset']['batch_size']
                
                
#                 # return
#             if epoch % exp_params['optimizer']['log_interval'] == 0:
#                 est_loss /= T
#                 sim_loss /= T
#                 sparsity /= T
#                 penalty /= T
#                 l1_loss /= T
    

        
#                 D_true = true_vals['D']
#                 D_W = model.W.cpu()
#                 H_est = torch.mean(variables['P'], dim=0).cpu()
#                 Z_est = variables['Z'].cpu()
#                 # assert(torch.linalg.det(H_est) != 0)
#                 D_est = torch.linalg.solve(H_est, Z_est.T) @ data / (data.shape[0]) # (K, N)
                
                    
#                 r2_w, r2 = compute_rsquared(D_W, D_est, variables, data)
#                 d_sim = compute_similarity(D_true, D_est)
#                 dw_sim = compute_similarity(D_true, D_W)
                
#                 Z_true = true_vals['Z']
#                 z_sim = compute_similarity(Z_true, Z_est)
                
#                 loss_vals['total_loss'].append(est_loss)
#                 loss_vals['similarity_loss'].append(sim_loss)
#                 loss_vals['l1_loss'].append(l1_loss)
#                 loss_vals['penalty'].append(penalty)
#                 loss_vals['sparsity'].append(sparsity)
#                 loss_vals['r2_W'].append(r2_w)
#                 # print(r2)
#                 loss_vals['r2_est'].append(r2)
#                 loss_vals['dw_sim'].append(dw_sim)
#                 loss_vals['dest_sim'].append(d_sim)
#                 loss_vals['latent_sim'].append(z_sim)
                
                
#                 pbar.set_postfix_str(f"Total Loss: {est_loss:.4f} | Similarity Loss: {sim_loss:.4f} | L1 Loss: {sparsity:.4f} | Penalty: {penalty:.4f} | R2_W: {r2_w:.4f} | R2_est: {r2:.4f} | D_sim: {d_sim:.4f} | Z_sim: {z_sim:.4f} | DW_sim: {dw_sim:.4f} | sparsity: {sparsity:.4f}")
        
#             # return  
                
#         return loss_vals





def run_ADMM_altmin(model, optimizer, data, variables, lagrange,
             dataloader, exp_params, true_vals, device,
             result_path=None, wandb_params=None, update_eta=False):
    """
    This function runs the KSM algorithm by tracking M and P. This is similar to ISTA.
    We call this Schedule 1.
    """
    loss_vals = {'total_loss': [], 'similarity_loss': [], 'l1_loss': [], 'penalty': [], 'sparsity': [], 'r2_W': [], 'r2_est': [], 'dw_sim': [], 'dest_sim':[], 'latent_sim': []}
    
    pbar = tqdm(range(exp_params['optimizer']['max_epochs']), desc='Training', position=0, leave=True)
    
    if update_eta:
        new_eta = optimizer.eta
        
    else:
        new_eta = 0
    T = variables['Z'].shape[0]
    with torch.no_grad():
        for epoch in pbar:
            
            est_loss = 0
            sim_loss = 0
            l1_loss = 0
            sparsity = 0
            penalty = 0
    
            for batch_idx, (input_batch, index) in enumerate(dataloader):
                
                # print(input_batch.to(device), index.shape)
                # return
                input_batch = input_batch.to(device)
                Z = variables['Z'][index]
                P = variables['P'][index]
                # M = lagrange['M'][index]
                
                # Here we track the lagrange multiplier
                M = model.W @ model.W.T / 2
                M = M.repeat(input_batch.shape[0], 1, 1)
                batch_variables = {'Z': Z, 'P': P}
                batch_lagrange = {'M': M}
                
                optimizer.set_values(input_batch, batch_variables, batch_lagrange)
                
                if update_eta:
                    new_eta = Z.shape[0] / power_method_svd(model.W.T, exp_params['device'])
                    # print(new_eta)
                    optimizer.update_eta(new_eta)
                


                optimizer.optim_variable('Z')
                optimizer.optim_param(norm=True)
                
                

                # Check for Nan and Inf
                
                if torch.isnan(optimizer.variables['Z']).any() or optimizer.variables['Z'].isinf().any():
                    if torch.isnan(optimizer.variables['Z']).any():
                        print('Nan detected')
                        print(batch_idx, epoch)
                        raise ValueError('Nan detected')
                    else:
                        print('Inf detected')
                        print(batch_idx, epoch)
                        raise ValueError('Inf detected')
                    
                
                    
                
                if torch.isnan(model.W).any() or model.W.isinf().any():
                    if torch.isnan(model.W).any():
                        print('param Nan detected')
                        print(batch_idx, epoch)
                        raise ValueError('param Nan detected')
                    else:
                        print('param Inf detected')
                        print(batch_idx, epoch)
                        raise ValueError('param Inf detected')
                    
                variables['Z'][index] = optimizer.variables['Z']
                variables['P'][index] = optimizer.variables['P']
                lagrange['M'][index] = optimizer.lags['M']
                
                
                
                # return
                
                if epoch % exp_params['optimizer']['log_interval'] == 0:
                    
                    loss_b, sim_b, l1_loss_b, penalty_b, sparsity_b = optimizer.get_loss()
                    
                    est_loss += loss_b.item() * exp_params['dataset']['batch_size']
                    sim_loss += sim_b.item() * exp_params['dataset']['batch_size']
                    l1_loss += l1_loss_b.item() * exp_params['dataset']['batch_size']
                    penalty += penalty_b.item() * exp_params['dataset']['batch_size']
                    sparsity += sparsity_b.item() * exp_params['dataset']['batch_size']
                
                
                # return
            if epoch % exp_params['optimizer']['log_interval'] == 0:
                est_loss /= T
                sim_loss /= T
                sparsity /= T
                penalty /= T
                l1_loss /= T
    
  
                D_true = true_vals['D']
                D_W = model.W.cpu()
                H_est = torch.mean(variables['P'], dim=0).cpu()
                Z_est = variables['Z'].cpu()
                D_est = torch.linalg.solve(H_est, Z_est.T) @ data / (data.shape[0]) # (K, N)
                
                    
                r2_w, r2 = compute_rsquared(D_W, D_est, variables, data)
                d_sim = compute_similarity(torch.abs(D_true), torch.abs(D_est))
                dw_sim = compute_similarity(torch.abs(D_true), torch.abs(D_W))
                
                Z_true = true_vals['Z']
                z_sim = compute_similarity(torch.abs(Z_true), torch.abs(Z_est))
                
                loss_vals['total_loss'].append(est_loss)
                loss_vals['similarity_loss'].append(sim_loss)
                loss_vals['l1_loss'].append(l1_loss)
                loss_vals['penalty'].append(penalty)
                loss_vals['sparsity'].append(sparsity)
                loss_vals['r2_W'].append(r2_w)
                loss_vals['r2_est'].append(r2)
                loss_vals['dw_sim'].append(dw_sim)
                loss_vals['dest_sim'].append(d_sim)
                loss_vals['latent_sim'].append(z_sim)
                
                
                pbar.set_postfix_str(f"Total Loss: {est_loss:.4f} | Similarity Loss: {sim_loss:.4f} | L1 Loss: {sparsity:.4f} | Penalty: {penalty:.4f} | R2_W: {r2_w:.4f} | R2_est: {r2:.4f} | D_sim: {d_sim:.4f} | Z_sim: {z_sim:.4f} | DW_sim: {dw_sim:.4f} | new_eta: {new_eta:.4f}")
        
            # return  
                
        return loss_vals



def run_ADMM_Ptrack(model, optimizer, data, variables, lagrange,
             dataloader, exp_params, device, true_vals=None, 
             result_path=None, wandb_params=None, update_eta=False, normW=False):
    """
    This function runs the KSM algorithm by tracking P. 
    We call this Schedule 2.
    """
    loss_vals = {'total_loss': [], 'similarity_loss': [], 'l1_loss': [], 'penalty': [], 'sparsity': [], 'r2_W': [], 'r2_est': [], 'dw_sim': [], 'dest_sim':[], 'latent_sim': []}
    
    pbar = tqdm(range(exp_params['optimizer']['max_epochs']), desc='Training', position=0, leave=True)
    
    if update_eta:
        new_eta = optimizer.eta
        
    else:
        new_eta = 0
    T = variables['Z'].shape[0]
    with torch.no_grad():
        for epoch in pbar:
            
            est_loss = 0
            sim_loss = 0
            l1_loss = 0
            sparsity = 0
            penalty = 0
    
            for batch_idx, (input_batch, index) in enumerate(dataloader):
                
                # print(input_batch.to(device), index.shape)
                # return
                input_batch = input_batch.to(device)
                Z = variables['Z'][index]
                P = variables['P'][index]
                M = lagrange['M'][index]
                
                # Here we track the lagrange multiplier
                # M = model.W @ model.W.T / 2
                # M = M.repeat(input_batch.shape[0], 1, 1)
                batch_variables = {'Z': Z, 'P': P}
                batch_lagrange = {'M': M}
                
                optimizer.set_values(input_batch, batch_variables, batch_lagrange)
                
                if update_eta:
                    new_eta = Z.shape[0] / power_method_svd(model.W.T, exp_params['device'])
                    # print(new_eta)
                    optimizer.update_eta(new_eta)
                


                optimizer.optim_variable('Z')
                optimizer.optim_param(norm=normW)
                optimizer.update_lags('M')
                

                # Check for Nan and Inf
                
                if torch.isnan(optimizer.variables['Z']).any() or optimizer.variables['Z'].isinf().any():
                    if torch.isnan(optimizer.variables['Z']).any():
                        print('Nan detected')
                        print(batch_idx, epoch)
                        raise ValueError('Nan detected')
                    else:
                        print('Inf detected')
                        print(batch_idx, epoch)
                        raise ValueError('Inf detected')
                    
                
                    
                
                if torch.isnan(model.W).any() or model.W.isinf().any():
                    if torch.isnan(model.W).any():
                        print('param Nan detected')
                        print(batch_idx, epoch)
                        raise ValueError('param Nan detected')
                    else:
                        print('param Inf detected')
                        print(batch_idx, epoch)
                        raise ValueError('param Inf detected')
                    
                variables['Z'][index] = optimizer.variables['Z']
                variables['P'][index] = optimizer.variables['P']
                lagrange['M'][index] = optimizer.lags['M']
                
                
                
                # return
                
                if epoch % exp_params['optimizer']['log_interval'] == 0:
                    
                    loss_b, sim_b, l1_loss_b, penalty_b, sparsity_b = optimizer.get_loss()
                    
                    est_loss += loss_b.item() * exp_params['dataset']['batch_size']
                    sim_loss += sim_b.item() * exp_params['dataset']['batch_size']
                    l1_loss += l1_loss_b.item() * exp_params['dataset']['batch_size']
                    penalty += penalty_b.item() * exp_params['dataset']['batch_size']
                    sparsity += sparsity_b.item() * exp_params['dataset']['batch_size']
                
                
                # return
            if epoch % exp_params['optimizer']['log_interval'] == 0:
                est_loss /= T
                sim_loss /= T
                sparsity /= T
                penalty /= T
                l1_loss /= T
    
  
                D_true = true_vals['D']
                D_W = model.W.cpu()
                H_est = torch.mean(variables['P'], dim=0).cpu()
                Z_est = variables['Z'].cpu()
                D_est = torch.linalg.solve(H_est, Z_est.T) @ data / (data.shape[0]) # (K, N)
                
                    
                r2_w, r2 = compute_rsquared(D_W, D_est, variables, data)
                
                d_sim = compute_similarity(torch.abs(D_true), torch.abs(D_est))
                dw_sim = compute_similarity(torch.abs(D_true), torch.abs(D_W))
                
                Z_true = true_vals['Z']
                z_sim = compute_similarity(torch.abs(Z_true), torch.abs(Z_est))
                
                loss_vals['total_loss'].append(est_loss)
                loss_vals['similarity_loss'].append(sim_loss)
                loss_vals['l1_loss'].append(l1_loss)
                loss_vals['penalty'].append(penalty)
                loss_vals['sparsity'].append(sparsity)
                loss_vals['r2_W'].append(r2_w)
                loss_vals['r2_est'].append(r2)
                loss_vals['dw_sim'].append(dw_sim)
                loss_vals['dest_sim'].append(d_sim)
                loss_vals['latent_sim'].append(z_sim)
                
                
                pbar.set_postfix_str(f"Total Loss: {est_loss:.4f} | Similarity Loss: {sim_loss:.4f} | L1 Loss: {sparsity:.4f} | Penalty: {penalty:.4f} | R2_W: {r2_w:.4f} | R2_est: {r2:.4f} | D_sim: {d_sim:.4f} | Z_sim: {z_sim:.4f} | DW_sim: {dw_sim:.4f} | new_eta: {new_eta:.4f}")
        
            # return  
                
        return loss_vals



def run_ADMM_notrack(model, optimizer, data, variables, lagrange,
             dataloader, exp_params, true_vals, device,
             result_path=None, wandb_params=None, update_eta=False, normW=False):
    """
    This class runs the KSM algorithm by tracking M and P. This is similar to ISTA. 
    We call this Schedule 3.
    """
    loss_vals = {'total_loss': [], 'similarity_loss': [], 'l1_loss': [], 'penalty': [], 'sparsity': [], 'r2_W': [], 'r2_est': [], 'dw_sim': [], 'dest_sim':[], 'latent_sim': []}
    
    pbar = tqdm(range(exp_params['optimizer']['max_epochs']), desc='Training', position=0, leave=True)
    
    if update_eta:
        new_eta = optimizer.eta
        
    else:
        new_eta = 0
    T = variables['Z'].shape[0]
    with torch.no_grad():
        for epoch in pbar:
            
            est_loss = 0
            sim_loss = 0
            l1_loss = 0
            sparsity = 0
            penalty = 0
    
            for batch_idx, (input_batch, index) in enumerate(dataloader):
                
                # print(input_batch.to(device), index.shape)
                # return
                input_batch = input_batch.to(device)
                Z = variables['Z'][index]
                P = variables['P'][index]
                M = lagrange['M'][index]
                
                # Here we track the lagrange multiplier
                # M = model.W @ model.W.T / 2
                # M = M.repeat(input_batch.shape[0], 1, 1)
                batch_variables = {'Z': Z, 'P': P}
                batch_lagrange = {'M': M}
                
                optimizer.set_values(input_batch, batch_variables, batch_lagrange)
                
                if update_eta:
                    factor = Z.shape[0] / 100
                    new_eta = factor / power_method_svd(model.W.T, exp_params['device'])
                    # print(new_eta)
                    optimizer.update_eta(new_eta)
                


                optimizer.optim_variable('Z')
                optimizer.optim_variable('P')
                optimizer.optim_param(norm=normW)
                optimizer.update_lags('M')
                

                # Check for Nan and Inf
                
                if torch.isnan(optimizer.variables['Z']).any() or optimizer.variables['Z'].isinf().any():
                    if torch.isnan(optimizer.variables['Z']).any():
                        print('Nan detected')
                        print(batch_idx, epoch)
                        raise ValueError('Nan detected')
                    else:
                        print('Inf detected')
                        print(batch_idx, epoch)
                        raise ValueError('Inf detected')
                    
                
                    
                
                if torch.isnan(model.W).any() or model.W.isinf().any():
                    if torch.isnan(model.W).any():
                        print('param Nan detected')
                        print(batch_idx, epoch)
                        raise ValueError('param Nan detected')
                    else:
                        print('param Inf detected')
                        print(batch_idx, epoch)
                        raise ValueError('param Inf detected')
                    
                variables['Z'][index] = optimizer.variables['Z']
                variables['P'][index] = optimizer.variables['P']
                lagrange['M'][index] = optimizer.lags['M']
                
                
                
                # return
                
                if epoch % exp_params['optimizer']['log_interval'] == 0:
                    
                    loss_b, sim_b, l1_loss_b, penalty_b, sparsity_b = optimizer.get_loss()
                    
                    est_loss += loss_b.item() * exp_params['dataset']['batch_size']
                    sim_loss += sim_b.item() * exp_params['dataset']['batch_size']
                    l1_loss += l1_loss_b.item() * exp_params['dataset']['batch_size']
                    penalty += penalty_b.item() * exp_params['dataset']['batch_size']
                    sparsity += sparsity_b.item() * exp_params['dataset']['batch_size']
                
                
                # return
            if epoch % exp_params['optimizer']['log_interval'] == 0:
                est_loss /= T
                sim_loss /= T
                sparsity /= T
                penalty /= T
                l1_loss /= T
    
  
                D_true = true_vals['D']
                D_W = model.W.cpu()
                H_est = torch.mean(variables['P'], dim=0).cpu()
                Z_est = variables['Z'].cpu()
                D_est = torch.linalg.solve(H_est, Z_est.T) @ data / (data.shape[0]) # (K, N)
                
                    
                r2_w, r2 = compute_rsquared(D_W, D_est, variables, data)
                d_sim = compute_similarity(torch.abs(D_true), torch.abs(D_est))
                dw_sim = compute_similarity(torch.abs(D_true), torch.abs(D_W))
                
                Z_true = true_vals['Z']
                z_sim = compute_similarity(torch.abs(Z_true), torch.abs(Z_est))
                
                loss_vals['total_loss'].append(est_loss)
                loss_vals['similarity_loss'].append(sim_loss)
                loss_vals['l1_loss'].append(l1_loss)
                loss_vals['penalty'].append(penalty)
                loss_vals['sparsity'].append(sparsity)
                loss_vals['r2_W'].append(r2_w)
                loss_vals['r2_est'].append(r2)
                loss_vals['dw_sim'].append(dw_sim)
                loss_vals['dest_sim'].append(d_sim)
                loss_vals['latent_sim'].append(z_sim)
                
                
                pbar.set_postfix_str(f"Total Loss: {est_loss:.4f} | Similarity Loss: {sim_loss:.4f} | L1 Loss: {sparsity:.4f} | Penalty: {penalty:.4f} | R2_W: {r2_w:.4f} | R2_est: {r2:.4f} | D_sim: {d_sim:.4f} | Z_sim: {z_sim:.4f} | DW_sim: {dw_sim:.4f} | new_eta: {new_eta:.4f}")
        
            # return  
                
        return loss_vals




def run_ADMM_kds_Ptrack(model, optimizer, data, variables, lagrange,
             dataloader, exp_params, device, true_vals=None, 
             result_path=None, wandb_params=None, update_eta=False, normW=False):
    """
    This class runs the KSM algorithm by tracking P for manifold learning, under schedules 2.
    """
    loss_vals = {'total_loss': [], 'similarity_loss': [], 'l1_loss': [], 'penalty': [], 'sparsity': [], 'r2_W': [], 'r2_est': [], 'dw_sim': [], 'dest_sim':[], 'latent_sim': []}
    
    pbar = tqdm(range(exp_params['optimizer']['max_epochs']), desc='Training', position=0, leave=True)
    
    if update_eta:
        new_eta = optimizer.eta
        
    else:
        new_eta = 0
    T = variables['Z'].shape[0]
    with torch.no_grad():
        for epoch in pbar:
            
            est_loss = 0
            sim_loss = 0
            penalty = 0
    
            for batch_idx, (input_batch, index) in enumerate(dataloader):
                
                # print(input_batch.to(device), index.shape)
                # return
                input_batch = input_batch.to(device)
                Z = variables['Z'][index]
                P = variables['P'][index]
                M = lagrange['M'][index]
                
                # Here we track the lagrange multiplier
                # M = model.W @ model.W.T / 2
                # M = M.repeat(input_batch.shape[0], 1, 1)
                batch_variables = {'Z': Z, 'P': P}
                batch_lagrange = {'M': M}
                
                optimizer.set_values(input_batch, batch_variables, batch_lagrange)
                
                if update_eta:
                    new_eta = 1 / power_method_svd(model.W.T, exp_params['device'])
                    # print(new_eta)
                    optimizer.update_eta(new_eta)
                


                optimizer.optim_variable('Z')
                optimizer.optim_param(norm=normW)
                optimizer.update_lags('M')
                

                # Check for Nan and Inf
                
                if torch.isnan(optimizer.variables['Z']).any() or optimizer.variables['Z'].isinf().any():
                    if torch.isnan(optimizer.variables['Z']).any():
                        print('Nan detected')
                        print(batch_idx, epoch)
                        raise ValueError('Nan detected')
                    else:
                        print('Inf detected')
                        print(batch_idx, epoch)
                        raise ValueError('Inf detected')
                    
                
                    
                
                if torch.isnan(model.W).any() or model.W.isinf().any():
                    if torch.isnan(model.W).any():
                        print('param Nan detected')
                        print(batch_idx, epoch)
                        raise ValueError('param Nan detected')
                    else:
                        print('param Inf detected')
                        print(batch_idx, epoch)
                        raise ValueError('param Inf detected')
                    
                variables['Z'][index] = optimizer.variables['Z']
                variables['P'][index] = optimizer.variables['P']
                lagrange['M'][index] = optimizer.lags['M']
                
                
                
                # return
                
                if epoch % exp_params['optimizer']['log_interval'] == 0:
                    
                    loss_b, sim_b, penalty_b = optimizer.get_loss()
                    
                    est_loss += loss_b.item() * exp_params['dataset']['batch_size']
                    sim_loss += sim_b.item() * exp_params['dataset']['batch_size']
                    # l1_loss += l1_loss_b.item() * exp_params['dataset']['batch_size']
                    penalty += penalty_b.item() * exp_params['dataset']['batch_size']
                    # sparsity += sparsity_b.item() * exp_params['dataset']['batch_size']
                
                
                # return
            if epoch % exp_params['optimizer']['log_interval'] == 0:
                est_loss /= T
                sim_loss /= T
                # sparsity /= T
                penalty /= T
                # l1_loss /= T
    
  
                
                D_W = model.W.cpu() 
                H_est = torch.mean(variables['P'], dim=0).cpu()
                Z_est = variables['Z'].cpu()
                D_est = (1 + model.omega) * torch.linalg.solve(H_est, Z_est.T) @ data / (data.shape[0]) # (K, N) 
                
                    
                r2_w, r2 = compute_rsquared(D_W, D_est, variables, data)
                
                if true_vals:
                    D_true = true_vals['D']
                    d_sim = compute_similarity(torch.abs(D_true), torch.abs(D_est))
                    dw_sim = compute_similarity(torch.abs(D_true), torch.abs(D_W))
                
                    Z_true = true_vals['Z']
                    z_sim = compute_similarity(torch.abs(Z_true), torch.abs(Z_est))
                    loss_vals['dw_sim'].append(dw_sim)
                    loss_vals['dest_sim'].append(d_sim)
                    loss_vals['latent_sim'].append(z_sim)
                    
                else:
                    d_sim = 0
                    dw_sim = 0
                    z_sim = 0
                
                
                loss_vals['total_loss'].append(est_loss)
                loss_vals['similarity_loss'].append(sim_loss)
                # loss_vals['l1_loss'].append(l1_loss)
                loss_vals['penalty'].append(penalty)
                # loss_vals['sparsity'].append(sparsity)
                loss_vals['r2_W'].append(r2_w)
                loss_vals['r2_est'].append(r2)
                
                
                pbar.set_postfix_str(f"Total Loss: {est_loss:.4f} | Similarity Loss: {sim_loss:.4f} | Penalty: {penalty:.4f} | R2_W: {r2_w:.4f} | R2_est: {r2:.4f} | D_sim: {d_sim:.4f} | Z_sim: {z_sim:.4f} | DW_sim: {dw_sim:.4f} | new_eta: {new_eta:.4f}")
        
            # return  
                
        return loss_vals
