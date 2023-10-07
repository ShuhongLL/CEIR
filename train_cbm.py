import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # make cuda deterministic

import torch
import random
import utils
import argparse
import datetime
import json
import numpy as np
import components.data_utils as data_utils
import components.similarity as similarity
import train_vae
from utils import set_seed

parser = argparse.ArgumentParser(description='Settings for creating CBM')

parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--concept_set", type=str, default=None, 
                    help="path to concept set name")
parser.add_argument("--backbone", type=str, default="clip_RN50", help="Which pretrained model to use as backbone")
parser.add_argument("--clip_name", type=str, default="ViT-B/16", help="Which CLIP model to use")

parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size used when saving model/CLIP activations")
parser.add_argument("--proj_batch_size", type=int, default=50000, help="Batch size to use when learning projection layer")

parser.add_argument("--gpt_version", type=str, default="gpt4", help="Which gpt version used for concept extraction")
parser.add_argument("--feature_layer", type=str, default='layer4', 
                    help="Which layer to collect activations from. Should be the name of second to last layer in the model")
parser.add_argument("--activation_dir", type=str, default='saved_activations', help="Save location for backbone and CLIP activations")
parser.add_argument("--save_dir", type=str, default='saved_models', help="Where to save trained models")
parser.add_argument("--save_concept_activation", type=bool, default=True, help="Whether save the generated concept activation vector")
parser.add_argument("--clip_cutoff", type=float, default=0.25, help="Concepts with smaller top5 clip activation will be deleted")
parser.add_argument("--proj_steps", type=int, default=1000, help="How many steps to train the projection layer for")
parser.add_argument("--interpretability_cutoff", type=float, default=0.45, help="concepts with smaller similarity to target concept will be deleted")
parser.add_argument("--print", type=bool, default=False, help="Print all concepts being deleted in this stage")

parser.add_argument('--train_vae', type=bool, default=True, help="Whether train the vae autoencoder")
parser.add_argument('--vae_train_set', type=str, default='both', help="Self-supervised training of vae on 'train' or 'val' or 'both' dataset")
parser.add_argument('--vae_hidden_dim', type=int, default=512)
parser.add_argument('--vae_latent_dim', type=int, default=256)
parser.add_argument('--vae_epochs', type=int, default=100)
parser.add_argument('--vae_batch_size', type=int, default=256)
parser.add_argument('--vae_learning_rate', type=float, default=5e-5)
parser.add_argument('--vae_save_interval', type=int, default=25, help="How many epoch interval to save VAE model")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--multiprocess', type=bool, default=False, help="Enable multiprocess of running kmean")
parser.add_argument('--save_vae', type=bool, default=True, help="Whether save the trained vae model")

def train_cbm_and_save(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if args.concept_set==None:
        args.concept_set = "data/concept_sets/{}_filtered.txt".format(args.dataset)
        
    similarity_fn = similarity.cos_similarity_cubed_single
    
    d_train = args.dataset + "_train"
    d_val = args.dataset + "_val"
    
    # get concept set
    cls_file = data_utils.LABEL_FILES[args.dataset]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
    if args.print:
        print(f"Dataset: {args.dataset}")
        print(f"Number of classes: {len(classes)}")
    args.num_cluster = len(classes)
    
    with open(args.concept_set) as f:
        concepts = f.read().split("\n")
    
    # save activations and get save_paths
    if args.print:
        print("Saving CLIP activation...")
    for d_probe in [d_train, d_val]:
        utils.save_activations(clip_name = args.clip_name, target_name = args.backbone, 
                               target_layers = [args.feature_layer], d_probe = d_probe,
                               concept_set = args.concept_set, batch_size = args.batch_size, 
                               device = args.device, pool_mode = "avg", save_dir = args.activation_dir)
        
    target_save_name, clip_save_name, text_save_name = utils.get_save_names(args.clip_name, args.backbone, 
                                            args.feature_layer, d_train, args.concept_set, "avg", args.activation_dir)
    val_target_save_name, val_clip_save_name, text_save_name =  utils.get_save_names(args.clip_name, args.backbone,
                                            args.feature_layer, d_val, args.concept_set, "avg", args.activation_dir)
    
    if args.print:
        print("CLIP activation saved.")
    
    # load features
    with torch.no_grad():
        target_features = torch.load(target_save_name, map_location="cpu").float()
        
        val_target_features = torch.load(val_target_save_name, map_location="cpu").float()
    
        image_features = torch.load(clip_save_name, map_location="cpu").float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        val_image_features = torch.load(val_clip_save_name, map_location="cpu").float()
        val_image_features /= torch.norm(val_image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()
        text_features /= torch.norm(text_features, dim=1, keepdim=True)
        
        clip_features = image_features @ text_features.T
        val_clip_features = val_image_features @ text_features.T

        del image_features, text_features, val_image_features
    
    if args.print:
        print("Filtering concept not activating highly...")
    # filter concepts not activating highly
    highest = torch.mean(torch.topk(clip_features, dim=0, k=5)[0], dim=0)
    
    if args.print:
        for i, concept in enumerate(concepts):
            if highest[i]<=args.clip_cutoff:
                print("Deleting {}, CLIP top5:{:.3f}".format(concept, highest[i]))
    concepts = [concepts[i] for i in range(len(concepts)) if highest[i]>args.clip_cutoff]
    
    # save memory by recalculating
    del clip_features
    with torch.no_grad():
        image_features = torch.load(clip_save_name, map_location="cpu").float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()[highest>args.clip_cutoff]
        text_features /= torch.norm(text_features, dim=1, keepdim=True)
    
        clip_features = image_features @ text_features.T
        del image_features, text_features
    
    val_clip_features = val_clip_features[:, highest>args.clip_cutoff]
    
    # learn projection layer
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts),
                                 bias=False).to(args.device)
    opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)
    
    indices = [ind for ind in range(len(target_features))]
    
    if args.print:
        print("Training projection layer...")
    best_val_loss = float("inf")
    best_step = 0
    best_weights = None
    proj_batch_size = min(args.proj_batch_size, len(target_features))
    for i in range(args.proj_steps):
        batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        outs = proj_layer(target_features[batch].to(args.device).detach())
        loss = -similarity_fn(clip_features[batch].to(args.device).detach(), outs)
        
        loss = torch.mean(loss)
        loss.backward()
        opt.step()
        if i % 50 == 0 or i == args.proj_steps-1:
            with torch.no_grad():
                val_output = proj_layer(val_target_features.to(args.device).detach())
                val_loss = -similarity_fn(val_clip_features.to(args.device).detach(), val_output)
                val_loss = torch.mean(val_loss)
            if i == 0:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
                print("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(best_step, -loss.cpu(),
                                                                                               -best_val_loss.cpu()))
                
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
            else: #stop if val loss starts increasing
                break
        opt.zero_grad()
        
    proj_layer.load_state_dict({"weight":best_weights})
    print("Best step:{}, Avg val similarity:{:.4f}".format(best_step, -best_val_loss.cpu()))
    
    # delete concepts that are not interpretable
    with torch.no_grad():
        outs = proj_layer(val_target_features.to(args.device).detach())
        sim = similarity_fn(val_clip_features.to(args.device).detach(), outs)
        interpretable = sim > args.interpretability_cutoff
        
    if args.print:
        for i, concept in enumerate(concepts):
            if sim[i]<=args.interpretability_cutoff:
                print("Deleting {}, Iterpretability:{:.3f}".format(concept, sim[i]))
    
    concepts = [concepts[i] for i in range(len(concepts)) if interpretable[i]]
    
    del clip_features, val_clip_features
    
    W_c = proj_layer.weight[interpretable]
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts), bias=False)
    proj_layer.load_state_dict({"weight":W_c})
    
    train_targets = data_utils.get_targets_only(d_train)
    val_targets = data_utils.get_targets_only(d_val)
    
    with torch.no_grad():
        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
        
        train_c -= train_mean
        train_c /= train_std
        
        train_y = torch.LongTensor(train_targets)
        # indexed_train_ds = IndexedTensorDataset(train_c, train_y)

        val_c -= train_mean
        val_c /= train_std
        val_y = torch.LongTensor(val_targets)

    base_save_path = (
        f"{args.save_dir}_{args.gpt_version}/"
        f"{args.dataset}/{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    )

    cbm_save_name = f"{base_save_path}/cbm_{args.backbone}"
    args.cbm_save_name = cbm_save_name
    args.base_save_path = base_save_path

    os.makedirs(cbm_save_name, exist_ok=False)
    torch.save(train_mean, os.path.join(cbm_save_name, "proj_mean.pt"))
    torch.save(train_std, os.path.join(cbm_save_name, "proj_std.pt"))
    torch.save(W_c, os.path.join(cbm_save_name ,"W_c.pt"))
    
    with open(os.path.join(cbm_save_name, "concepts.txt"), 'w') as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write('\n'+concept)
    
    with open(os.path.join(cbm_save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.save_concept_activation:
        np.save(f'{cbm_save_name}/train_concepts.npy', train_c)
        np.save(f'{cbm_save_name}/train_y.npy', train_y)
        np.save(f'{cbm_save_name}/val_concepts.npy', val_c)
        np.save(f'{cbm_save_name}/val_y.npy', val_y)
        print(f"concept_load_path: {cbm_save_name}")

    del train_targets, val_targets
    del train_mean, train_std

    return (train_c.cpu().numpy(),
            train_y.cpu().numpy(),
            val_c.cpu().numpy(),
            val_y.cpu().numpy())
    
if __name__=='__main__':
    args = parser.parse_args()
    set_seed(args.seed)
    latent_concepts = train_cbm_and_save(args)

    print(f"base_save_path: {args.base_save_path}\n")

    if args.train_vae:
        config = {}
        config['dataset'] = args.dataset
        config['num_cluster'] = args.num_cluster
        config['train_set'] = args.vae_train_set
        config['hidden_dim'] = args.vae_hidden_dim
        config['latent_dim'] = args.vae_latent_dim
        config['epochs'] = args.vae_epochs
        config['batch_size'] = args.vae_batch_size
        config['learning_rate'] = args.vae_learning_rate
        config['enable_dropout'] = False
        config['enable_batchnorm'] = True
        config['save_interval'] = args.vae_save_interval
        config['base_save_path'] = args.base_save_path
        config['seed'] = args.seed
        config['device'] = args.device
        config['multiprocess'] = args.multiprocess
        config['save_vae'] = args.save_vae
        config['backbone'] = args.backbone

        print()

        if args.print:
            print("Training VAE...")
        train_vae.train(config, latent_concepts)
