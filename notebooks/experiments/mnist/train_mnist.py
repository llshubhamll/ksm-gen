import os
import numpy as np
import matplotlib.pyplot as plt
import hydra
import torch
import pickle


from torchvision.utils import make_grid
from datetime import datetime
from src.models import KSM_objective
from omegaconf import DictConfig, OmegaConf 
from src.datasets import get_mnist_dataset
from pathlib import Path
from tqdm import tqdm
from src.optimizers import optimizationADMM_proximal
from src.utils import power_method_svd
from src.visualizations import visualize_dense_dictionary
from src.run_utils import compute_run_metrics
import wandb



def load_data(exp_params):
    """
    Loads Mnist data
    """
    data_path = Path.cwd().parents[0]/'data/data_cache'
    print(data_path)
    X, y, *_ = get_mnist_dataset(class_list=exp_params['dataset']['class_list'], 
                                 fraction=exp_params['dataset']['fraction'], datadir=data_path)
    # print(data_path)
    
    return X, y




def initializeTensors(exp_params, n_samples, Zinit=None, Pinit=None, Minit=None, seed=42, device='cpu'):
    torch.manual_seed(seed)
    if Zinit is None:
        Z = torch.randn(size=(n_samples, exp_params['model']['K']), dtype=torch.float32, requires_grad=False)
    else:
        Z = Zinit.clone()
        
        
    if Pinit is None:    
        Z_t = Z.unsqueeze(-1)
        V = Z_t @ Z_t.transpose(-1, -2)
        P = V + exp_params['model']['omega'] * torch.eye(exp_params['model']['K'], dtype=torch.float32, requires_grad=False, device=device)
    else:
        P = Pinit.clone()
        
        
    if Minit is None:        
        M_t = torch.randn(size=P.shape, requires_grad=False, dtype=torch.float32)
        M = M_t @ M_t.transpose(-1, -2)
        M = M / torch.linalg.norm(M, dim=(1, 2), keepdims=True)
    else:
        M = Minit.clone()
        
        
    # print("Initialized M")
    
    variables = {'Z': Z, 'P': P}
    lagrange = {'M': M}
    
    return variables, lagrange



def visualize_reconst_batch(input_batch, latents, D_est, latent_pos, fig=None):
    
    if fig is None:
        fig, axs = plt.subplots(1, 3)
        
        
        if len(input_batch.shape) == 2:
            # input_batch = input_batch.T
            input_batch = input_batch.reshape(-1, 1, 28, 28)
        
        Zs = latents[latent_pos]
        # print("Zs", Zs.shape)
        
        reconst_data = Zs @ D_est
        
        input_matrix = make_grid(input_batch, nrow=1, padding=2, normalize=True).numpy().transpose(1, 2, 0)
        nrows = 20
        ncols = Zs.shape[1] // nrows
        Zs = Zs.reshape(-1, 1, nrows, ncols)
        Z_matrix = make_grid(Zs.cpu(), nrow=1, padding=2, normalize=True).numpy().transpose(1, 2, 0)
        reconst_data = reconst_data.reshape(-1, 1, 28, 28)
        reconst_matrix = make_grid(reconst_data.cpu(), nrow=1, padding=2, normalize=True).numpy().transpose(1, 2, 0)
        
        data_matrices = [input_matrix, Z_matrix, reconst_matrix]
        labels = ['Input', 'Latent', 'Reconstruction']
        for i, ax in enumerate(axs):
            ax.imshow(data_matrices[i], cmap='gray'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(labels[i])
            
            
        return fig
        
    


@hydra.main(version_base=None, config_path="../../../configs", config_name='ksm_mnist.yaml')
def main(cfg:DictConfig):
    
    exp_params = OmegaConf.to_container(cfg, resolve=True)
    
    np.random.seed(42)
    X, y = load_data(exp_params)
    data = X.reshape(X.shape[0], -1)
    data = np.float32(data)
    # data = data.T
    # print(data[0].reshape(28, 28))
    # print(data.shape)
    # return
    # Create random samples for visualization
    numOfImages = 5
    random_samples = np.random.choice(data.shape[0], numOfImages, replace=False)
    viz_batch = data[random_samples]
    # print("data", data.shape)
    # print(data.shape)
    # exit()
    
    exp_params['n_batches'] = data.shape[0] // exp_params['dataset']['batch_size']
    runtime = {'T': data.shape[0], 'N': data.shape[1]}
    exp_params['runtime'] = runtime
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_params['device'] = device
    
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(data).float(), torch.arange(data.shape[0]), torch.from_numpy(y))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=exp_params['dataset']['batch_size'], shuffle=True)
    
    
    current_dir = Path(hydra.utils.get_original_cwd())
    
    # Load initial values
    # ==================== #
    print("Loading initial values")
    print("======================")
    intial_dir = current_dir.parents[2] / 'models' / 'ksm_mnist'
    # initial_filename = 'mbdl_170epochs_codes_K400_lam1.00e-01.pkl'
    initial_filename = 'mbdl_kmeans_10k.pkl'
    
    with open(intial_dir / initial_filename, 'rb') as f:
        initial_dict = pickle.load(f)
        dict_obj = initial_dict['dict']
        initial_codes = initial_dict['codes']
        initial_dict = torch.from_numpy(dict_obj).float().to(device)
        initial_codes = torch.from_numpy(initial_codes).float().to(device)
        
        
    
    # Define objective
    # ==================== #
    print("Defining objective")
    print("===================")
    model = KSM_objective(exp_params, W_init=initial_dict)
    

    
    # Define WANDB logging names
    # ==================== #
    
    # project_name = f"ksm_{exp_params['dataset']['name']}"
    project_name = "ksm_mnist"
    # project_name = 'ksm_test'
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = f"{project_name}_{timestamp}"
    
    
    
    # Initialize tensors
    # ==================== #
    print("Initialize tensors")
    print("===================")    
    
    M_init = initial_dict @ initial_dict.T / 2
    M_init = M_init.repeat(runtime['T'], 1, 1).to(device)
    Z_init = initial_codes.clone().to(device)
    variables, lagrange = initializeTensors(exp_params, runtime['T'], Zinit=Z_init, Minit=M_init, device=device)
    eta = exp_params['dataset']['batch_size'] / power_method_svd(model.W.T, exp_params['device'])
    optimizer = optimizationADMM_proximal(exp_params, model, eta=eta)
    run_metrics = compute_run_metrics(exp_params, torch.from_numpy(data).to(device))
    
    data = torch.from_numpy(data).to(device)
    
    pbar = tqdm(range(exp_params['optimizer']['max_epochs']), desc='Training', position=0, leave=True)    
    # return
    with wandb.init(project=project_name, name=exp_name, config=exp_params):
        
        loss_vals = {
            'sim_term': [],
            'sparsity': [],
            'total': [],
            'lagrangian': [],
            'reconstruction_loss': [],
            'constraint_val': [],
        }
        wandb.define_metric('Train/epoch')
        wandb.define_metric('Train/*', step_metric='Train/epoch')
        
        # Save the model

        timestamp = Path(f"{datetime.now().strftime('%m-%d')}") / Path(f"{datetime.now().strftime('%H-%M')}")
        result_folder = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) 
        # result_folder = current_dir / timestamp
        # os.makedirs(result_folder, exist_ok=True)
        


        best_loss = np.inf
        loss_vals = {'total_loss': [], 'similarity_loss': [], 'l1_loss': [], 'penalty': [], 'sparsity': [], 'r2_W': [], 'r2_est': [], 'dw_sim': [], 'dest_sim':[], 'latent_sim': []}
        
        if exp_params['update_eta']:
            new_eta = optimizer.eta
                    
        else:
            new_eta = 0
        
        T = variables['Z'].shape[0]

            
        # return   
        with torch.no_grad():
            for epoch in pbar:
            
                est_loss = 0
                sim_loss = 0
                l1_loss = 0
                sparsity = 0
                penalty = 0
    
                for batch_idx, (input_batch, index, _) in enumerate(dataloader):
                    
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
                    
                    if exp_params['update_eta']:
                        new_eta = Z.shape[0] / power_method_svd(model.W.T, exp_params['device'])
                        # print(new_eta)
                        optimizer.update_eta(new_eta)
                    


                    optimizer.optim_variable('Z')
                    optimizer.optim_param(norm=True)
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
                
                    if epoch % exp_params['optimizer']['log_interval'] == 0 or epoch == exp_params['optimizer']['max_epochs'] - 1:
                        
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
        
    
                    # D_true = 
                    D_W = model.W
                    H_est = torch.mean(variables['P'], dim=0)
                    Z_est = variables['Z']
                    D_est = torch.linalg.solve(H_est, Z_est.T) @ data / (data.shape[0]) # (K, N)
                    
                        
                    r2, r2_w = run_metrics.compute_r2(model, variables)
                    # d_sim = compute_similarity(torch.abs(D_true), torch.abs(D_est))
                    # dw_sim = compute_similarity(torch.abs(D_true), torch.abs(D_W))
                    
                    # Z_true = true_vals['Z']
                    # z_sim = compute_similarity(torch.abs(Z_true), torch.abs(Z_est))
                    
                    wandb.log({
                        'Train/total_loss': est_loss,
                        'Train/similarity_loss': sim_loss,
                        'Train/l1_loss': l1_loss,
                        'Train/penalty': penalty,
                        'Train/sparsity': sparsity,
                        'Train/r2_W': r2_w,
                        'Train/r2_est': r2,
                        # 'Train/dw_sim': dw_sim,
                        # 'Train/dest_sim': d_sim,
                        # 'Train/latent_sim': z_sim,
                        'Train/new_eta': new_eta,
                        'Train/epoch': epoch,
                    })
                    
                    loss_vals['total_loss'].append(est_loss)
                    loss_vals['similarity_loss'].append(sim_loss)
                    loss_vals['l1_loss'].append(l1_loss)
                    loss_vals['penalty'].append(penalty)
                    loss_vals['sparsity'].append(sparsity)
                    loss_vals['r2_W'].append(r2_w)
                    loss_vals['r2_est'].append(r2)
                    # loss_vals['dw_sim'].append(dw_sim)
                    # loss_vals['dest_sim'].append(d_sim)
                    # loss_vals['latent_sim'].append(z_sim)
                    
                    
                    pbar.set_postfix_str(f"Total Loss: {est_loss:.4f} | Similarity Loss: {sim_loss:.4f} | L1 Loss: {sparsity:.4f} | Penalty: {penalty:.4f} | R2_W: {r2_w:.4f} | R2_est: {r2:.4f} | new_eta: {new_eta:.4f}")
                    
                    result_file = f"ksm_epoch{epoch}.pkl"
                    result_path = result_folder / result_file
                    save_data = {'model': model, 'exp_params': exp_params, 'loss_dict': loss_vals, 'Z': variables['Z'].cpu(), 'data': data.cpu()}
                    with open(result_path, 'wb') as f:
                        pickle.dump(save_data, f)
                    # best_loss = est_loss
            
                
                
                if (epoch % exp_params['optimizer']['vis_interval'] == 0) or (epoch == exp_params['optimizer']['max_epochs'] - 1):
                    
                    D_viz = D_W[:100] # Visualize only the first 100 atoms
                    # print(torch.mean(torch.linalg.norm(D_viz, dim=1)))
                    Dest_viz = D_est[:100]
                    Dest_viz = Dest_viz / torch.linalg.norm(Dest_viz, dim=1, keepdims=True)
                    
                    fig1 = visualize_dense_dictionary(D_viz.T.cpu())
                    fig2 = visualize_dense_dictionary(Dest_viz.T.cpu())
                    fig3 = visualize_reconst_batch(torch.from_numpy(viz_batch), variables['Z'], D_W, random_samples)
                    
                    wandb.log({
                        'Train/W': fig1,
                        'Train/dictionary': fig2, 
                        'Train/reconstruction': fig3,
                    })  
                    plt.close('all')
                    
                
                # Save best loss
                

                
            # pbar.set_description(f"Epoch: {epoch}/{exp_params['optimizer']['max_epochs']}")
            # break
        
        
    
    
    # print(result_path)
    # # exit()
    # save_data = {'dl_class': dl, 'exp_params': exp_params, 'loss_dict': loss_vals}
    
    # with open(result_path, 'wb') as f:
    #     pickle.dump(save_data, f)
        
        
        
        
if __name__ == '__main__':
    main()
                
                
        
        
        
        
        
    
    
    
    
    
    
    
    