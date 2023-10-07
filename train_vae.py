import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # make cuda deterministic 

import torch
import torch.optim as optim
import numpy as np
import argparse
import multiprocessing
import components.data_utils as data_utils
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from components.vae import VAE_base, vae_loss
from utils import set_seed, get_vae_embedding, get_vae_path, run_kmean

def train(config, concepts=None):
    set_seed(config['seed'])

    if concepts:
        x_train, y_train, x_test, y_test = concepts
    else:
        x_train = np.load(f"{config['concept_load_path']}/train_concepts.npy")
        y_train = np.load(f"{config['concept_load_path']}/train_y.npy")
        x_test = np.load(f"{config['concept_load_path']}/val_concepts.npy")
        y_test = np.load(f"{config['concept_load_path']}/val_y.npy")

    x_train = np.reshape(x_train, (len(x_train), -1))
    x_test = np.reshape(x_test, (len(x_test), -1))

    lock = multiprocessing.Lock()
    device = torch.device(config['device'])

    if config['train_set'] == 'train':
        dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    elif config['train_set'] == 'val':
        dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    elif config['train_set'] == 'both':
        x_combine = np.concatenate((x_train, x_test), axis=0)
        y_combine = np.concatenate((y_train, y_test), axis=0)
        dataset = TensorDataset(torch.from_numpy(x_combine), torch.from_numpy(y_combine))
    else:
        raise ValueError(f"Invalid training set: {config['train_set']}")

    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    input_dim = x_train.shape[1]

    vae = VAE_base(input_dim, config['hidden_dim'], config['latent_dim'],
                   dropout=config['enable_dropout'], batch_norm=config['enable_batchnorm'])
    print(vae)
    
    vae = vae.double().to(device)
    optimizer = optim.Adam(vae.parameters(), lr=config['learning_rate'])
    # torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1)

    # Training loop
    for epoch in tqdm(range(config['epochs'])):
        vae.train()
        total_loss = 0
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device).double()
            optimizer.zero_grad()
            recon_data, mu, logvar = vae(data)
            loss = vae_loss(recon_data, data, mu, logvar)

            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {total_loss/len(data_loader.dataset):.4f}")
        if (epoch+1) % config['save_interval'] == 0:
            if config['save_vae']:
                os.makedirs(f"{config['base_save_path']}/vae", exist_ok=True)
                vae_model_path = get_vae_path(config['base_save_path'], config['train_set'], config['latent_dim'], 
                                            epoch+1, config['enable_dropout'], config['enable_batchnorm'])
                torch.save(vae.state_dict(), vae_model_path)
            
            with torch.no_grad():
                embeddings, labels = get_vae_embedding(vae, test_loader, device)

            # pool.apply_async(run_kmean, args=(lock, config, f"[{epoch+1}/{config['epochs']}]", embeddings, labels))
            # Fit K-means clustering to the embeddings
            if config['multiprocess']:
                process = multiprocessing.Process(target=run_kmean, \
                    args=(lock, config, f"[{epoch+1}/{config['epochs']}]", embeddings, labels))
                process.start()
            else:
                run_kmean(lock, config, f"[{epoch+1}/{config['epochs']}]", embeddings, labels)

            # run_kmean(lock, config, f"[{epoch+1}/{config['epochs']}]", embeddings, labels)       
    # Save the trained model
    # torch.save(vae.state_dict(), f"./vae_models/{config['dataset']}_vae_{config['train_set']}_model.pth")


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--concept_load_path', type=str, help="The path of the saved concept latent embedding")
    parser.add_argument('--base_save_path', type=str, help="The root path of the saved cbm and vae models")
    parser.add_argument('--vae_train_set', type=str, default='both', help="Self-supervised training of vae on 'train' or 'val' or 'both' dataset")
    parser.add_argument('--vae_hidden_dim', type=int, default=512)
    parser.add_argument('--vae_latent_dim', type=int, default=256)
    parser.add_argument('--vae_epochs', type=int, default=100)
    parser.add_argument('--vae_batch_size', type=int, default=256)
    parser.add_argument('--vae_learning_rate', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--multiprocess', type=bool, default=True)
    parser.add_argument('--save_vae', type=bool, default=True)

    args = parser.parse_args()
    classes = data_utils.get_classes(args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"Number of clusters: {len(classes)}")

    config = {}
    config['dataset'] = args.dataset
    config['concept_load_path'] = args.concept_load_path
    config['num_cluster'] = len(classes)
    config['train_set'] = args.vae_train_set
    config['hidden_dim'] = args.vae_hidden_dim
    config['latent_dim'] = args.vae_latent_dim
    config['epochs'] = args.vae_epochs
    config['batch_size'] = args.vae_batch_size
    config['learning_rate'] = args.vae_learning_rate
    config['seed'] = args.seed
    config['device'] = args.device
    config['save_interval'] = 25
    config['base_save_path'] = args.base_save_path
    config['enable_dropout'] = False
    config['enable_batchnorm'] = True
    config['multiprocess'] = args.multiprocess
    config['save_vae'] = args.save_vae

    print(config)
    print()

    train(config)
