"""
To eval VAE separately, you need to save the concept activation vectors during
the training cbm. set `train_cbm.py --save_concept_activation True`
"""

import os
import sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path.append('../')

import torch
import numpy as np
import argparse
import multiprocessing
import components.data_utils as data_utils
from torch.utils.data import TensorDataset, DataLoader
from components.vae import VAE_base
from utils import set_seed, get_vae_path, get_vae_embedding, run_kmean

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--concept_load_path', type=str, help="The path of the saved concept latent embedding")
    parser.add_argument('--base_save_path', type=str, help="The root path of the saved cbm and vae models")
    parser.add_argument('--vae_train_set', type=str, default='both', help="Self-supervised training of vae on 'train' or 'val' or 'both' dataset")
    parser.add_argument('--vae_hidden_dim', type=int, default=512)
    parser.add_argument('--vae_latent_dim', type=int, default=256)
    parser.add_argument('--eval_epochs', type=int, nargs='+', default=[50, 100, 150], help="select the list of epochs where the models were trained")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    set_seed(args.seed)
    classes = data_utils.get_classes(args.dataset)

    x_test = np.load(f"{args.concept_load_path}/val_concepts.npy")
    y_test = np.load(f"{args.concept_load_path}/val_y.npy")

    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    device = torch.device(args.device)
    lock = multiprocessing.Lock()

    for epoch in args.eval_epochs:
        configs = {}
        configs['dataset'] = args.dataset
        configs['concept_load_path'] = args.concept_load_path
        configs['base_save_path'] = args.base_save_path
        configs['num_cluster'] = len(classes)
        configs['train_set'] = args.vae_train_set
        configs['input_dim'] = x_test.shape[1]
        configs['hidden_dim'] = args.vae_hidden_dim
        configs['latent_dim'] = args.vae_latent_dim
        configs['enable_dropout'] = False
        configs['enable_batchnorm'] = True
        configs['seed'] = args.seed

        vae_model_path = get_vae_path(configs['base_save_path'], configs['train_set'], configs['latent_dim'],
                                      epoch, configs['enable_dropout'], configs['enable_batchnorm'])

        vae = VAE_base(configs['input_dim'], configs['hidden_dim'], configs['latent_dim'],
                        configs['enable_dropout'], configs['enable_batchnorm'])
        
        vae.load_state_dict(torch.load(vae_model_path))
        vae.eval()

        with torch.no_grad():
            embeddings, labels = get_vae_embedding(vae, test_loader, device)
            
        # Fit K-means clustering to the embeddings
        if args.multiprocess:
            process = multiprocessing.Process(target=run_kmean, \
                args=(lock, configs, f"[{epoch}]", embeddings, labels))
            process.start()
        else:
            run_kmean(lock, configs, f"[{epoch}]", embeddings, labels)
