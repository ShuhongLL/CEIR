import os
import math
import torch
import clip
import lap
import random
import json
import numpy as np
import components.cbm as cbm
import components.data_utils as data_utils
from tqdm import tqdm
from joblib import parallel_backend
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from components.vae import VAE_base

PM_SUFFIX = {"max":"_max", "avg":""}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(mode=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def save_target_activations(target_model, dataset, save_name, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg'):
    """
    save_name: save_file path, should include {} which will be formatted by layer names
    """
    _make_save_dir(save_name)
    save_names = {}    
    for target_layer in target_layers:
        save_names[target_layer] = save_name.format(target_layer)
        
    if _all_saved(save_names):
        return
    
    all_features = {target_layer:[] for target_layer in target_layers}
    
    hooks = {}
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
        hooks[target_layer] = eval(command)
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = target_model(images.to(device))
    
    for target_layer in target_layers:
        torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
        hooks[target_layer].remove()
    #free memory
    del all_features
    # torch.cuda.empty_cache()
    return

def save_clip_image_features(model, dataset, save_name, batch_size=1000 , device = "cuda"):
    _make_save_dir(save_name)
    all_features = []
    
    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = model.encode_image(images.to(device))
            all_features.append(features.cpu())
    torch.save(torch.cat(all_features), save_name)
    #free memory
    del all_features
    # torch.cuda.empty_cache()
    return

def save_clip_text_features(model, text, save_name, batch_size=1000):
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)
    del text_features
    # torch.cuda.empty_cache()
    return

def save_activations(clip_name, target_name, target_layers, d_probe, 
                     concept_set, batch_size, device, pool_mode, save_dir):
    target_save_name, clip_save_name, text_save_name = get_save_names(clip_name, target_name, 
                                                                    "{}", d_probe, concept_set, 
                                                                      pool_mode, save_dir)
    save_names = {"clip": clip_save_name, "text": text_save_name}
    for target_layer in target_layers:
        save_names[target_layer] = target_save_name.format(target_layer)
        
    if _all_saved(save_names):
        return
    
    clip_model, clip_preprocess = clip.load(clip_name, device=device)
    
    if target_name.startswith("clip_"):
        target_model, target_preprocess = clip.load(target_name[5:], device=device)
    else:
        target_model, target_preprocess = data_utils.get_target_model(target_name, device)
    #setup data
    data_c = data_utils.get_data(d_probe, clip_preprocess)
    data_t = data_utils.get_data(d_probe, target_preprocess)

    with open(concept_set, 'r') as f: 
        words = (f.read()).split('\n')
    print("Tokenizing concepts...")
    text = clip.tokenize(["{}".format(word) for word in words]).to(device)
    
    save_clip_text_features(clip_model, text, text_save_name, batch_size)
    save_clip_image_features(clip_model, data_c, clip_save_name, batch_size, device)
    if target_name.startswith("clip_"):
        save_clip_image_features(target_model, data_t, target_save_name, batch_size, device)
    else:
        save_target_activations(target_model, data_t, target_save_name, target_layers,
                                batch_size, device, pool_mode)
    return
    
def get_similarity_from_activations(target_save_name, clip_save_name, text_save_name, similarity_fn, 
                                   return_target_feats=True):
    image_features = torch.load(clip_save_name)
    text_features = torch.load(text_save_name)
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True).float()
        text_features /= text_features.norm(dim=-1, keepdim=True).float()
        clip_feats = (image_features @ text_features.T)
    del image_features, text_features
    # torch.cuda.empty_cache()
    
    target_feats = torch.load(target_save_name)
    similarity = similarity_fn(clip_feats, target_feats)
    
    del clip_feats
    # torch.cuda.empty_cache()
    
    if return_target_feats:
        return similarity, target_feats
    else:
        del target_feats
        # torch.cuda.empty_cache()
        return similarity
    
def get_activation(outputs, mode):
    '''
    mode: how to pool activations: one of avg, max
    for fc neurons does no pooling
    '''
    if mode=='avg':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output.mean(dim=[2,3]).detach().cpu())
            elif len(output.shape)==2:
                outputs.append(output.detach().cpu())
    elif mode=='max':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output.amax(dim=[2,3]).detach().cpu())
            elif len(output.shape)==2:
                outputs.append(output.detach().cpu())
    return hook

    
def get_save_names(clip_name, target_name, target_layer, d_probe, concept_set, pool_mode, save_dir):
    if target_name.startswith("clip_"):
        target_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, target_name.replace('/', ''))
    else:
        target_save_name = "{}/{}_{}_{}{}.pt".format(save_dir, d_probe, target_name, target_layer, 
                                                     PM_SUFFIX[pool_mode])
    clip_save_name = "{}/{}_clip_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
    concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
    text_save_name = "{}/{}_{}.pt".format(save_dir, concept_set_name, clip_name.replace('/', ''))
    
    return target_save_name, clip_save_name, text_save_name

    
def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return

def get_accuracy_cbm(model, dataset, device, batch_size=250, num_workers=2):
    correct = 0
    total = 0
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers,
                                           pin_memory=True)):
        with torch.no_grad():
            #outs = target_model(images.to(device))
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            correct += torch.sum(pred.cpu()==labels)
            total += len(labels)
    return correct/total

def get_preds_cbm(model, dataset, device, batch_size=250, num_workers=2):
    preds = []
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers,
                                           pin_memory=True)):
        with torch.no_grad():
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    return preds

def get_concept_act_by_pred(model, dataset, device):
    preds = []
    concept_acts = []
    for images, labels in tqdm(DataLoader(dataset, 500, num_workers=8, pin_memory=True)):
        with torch.no_grad():
            outs, concept_act = model(images.to(device))
            concept_acts.append(concept_act.cpu())
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    concept_acts = torch.cat(concept_acts, dim=0)
    concept_acts_by_pred=[]
    for i in range(torch.max(pred)+1):
        concept_acts_by_pred.append(torch.mean(concept_acts[preds==i], dim=0))
    concept_acts_by_pred = torch.stack(concept_acts_by_pred, dim=0)
    return concept_acts_by_pred

def get_vae_embedding(vae, test_loader, device):
    emds = []
    labels = []
    vae = vae.to(device).double()
    with torch.no_grad():
        for batch_idx, (data, y) in enumerate(test_loader):
            data = data.to(device).double()
            recon_data, mu, logvar = vae(data)
            emds.append(vae.get_z().detach().cpu().numpy())
            labels.append(y)
    embeddings = np.concatenate(emds, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embeddings, labels

def linear_assignment(cost_matrix):
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i], i] for i in x if i >= 0])

def contingency_matrix(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    return w

def acc_from_contingency(w):
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / np.sum(w)

def binom(n, k):
    if k < 0 or n < k:
        return 0
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))

def adjusted_rand_index_from_contingency(w):
    sum_comb = sum(sum(binom(nij, 2) for nij in row) for row in w)
    a = sum(binom(ni, 2) for ni in np.sum(w, axis=1))
    b = sum(binom(nj, 2) for nj in np.sum(w, axis=0))
    expected_index = a * b / binom(np.sum(w), 2)
    max_index = (a + b) / 2
    return (sum_comb - expected_index) / (max_index - expected_index)

def normalized_mutual_info_from_contingency(w):
    pi = np.sum(w, axis=1)
    pj = np.sum(w, axis=0)
    contingency_norm = w / np.sum(w)
    pi_norm = pi / np.sum(pi)
    pj_norm = pj / np.sum(pj)

    outer = np.outer(pi_norm, pj_norm)
    nnz = np.logical_and(contingency_norm != 0, outer != 0)

    mi = np.sum(contingency_norm[nnz] * np.log(contingency_norm[nnz] / outer[nnz]))
    h1 = -np.sum(pi_norm[pi_norm > 0] * np.log(pi_norm[pi_norm > 0]))
    h2 = -np.sum(pj_norm[pj_norm > 0] * np.log(pj_norm[pj_norm > 0]))

    return 2 * mi / (h1 + h2)

def run_kmean(lock, config, epoch_tag, embeddings, labels, dump_log=True):
    with parallel_backend('threading', n_jobs=1):
        kmeans = KMeans(n_clusters=config['num_cluster'], random_state=config['seed'])
        predicted_labels = kmeans.fit_predict(embeddings)
    w = contingency_matrix(labels, predicted_labels)
    accuracy = acc_from_contingency(w)
    ari = adjusted_rand_index_from_contingency(w)
    nmi = normalized_mutual_info_from_contingency(w)
    print()
    print(f"Clustering: Epoch={epoch_tag}, ACC={accuracy}, ARI={ari}, NMI={nmi}\n")
    if config['multiprocess']:
        lock.acquire()
    try:
        if dump_log:
            with open(f"{config['base_save_path']}/cluster_accuracies.txt", "a") as f:
                f.write(json.dumps(config))
                f.write("\n")
                f.write(f"Clustering: Epoch={epoch_tag}, ACC={accuracy}, ARI={ari}, NMI={nmi}\n")
                f.write("\n")
                f.flush()
    except Exception as e:
        print(f"Error in process: {e}")
    finally:
        # Release the lock after writing to the file
        if config['multiprocess']:
            lock.release()

def get_vae_path(base_save_path, train_set, latent_dim, epoch, enable_dropout=False, enable_batchnorm=True):
    vae_filename = (
        f"{base_save_path}/vae/{train_set}_z{latent_dim}"
        f"{'_dp' if enable_dropout else ''}"
        f"{'_bn' if enable_batchnorm else ''}"
        f"_e{epoch}.pth"
    )

    return vae_filename

def load_cbm_pipline(concept_load_path, device='cuda'):
    with open(os.path.join(concept_load_path, "args.txt"), "r") as f:
        args = json.load(f)
    _, target_preprocess = data_utils.get_target_model(args["backbone"], device)
    model = cbm.load_cbm(concept_load_path, device)
    return model, target_preprocess

def load_vae(vae_load_path, input_dim, hidden_dim, latent_dim):
    vae = VAE_base(input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, dropout=False, batch_norm=True)
    vae.load_state_dict(torch.load(vae_load_path))
    vae.eval()
    return vae
