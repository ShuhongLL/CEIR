import os
import sys
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # make cuda deterministic
os.environ["OPENBLAS_NUM_THREADS"] = "8" # you might modify this.
sys.path.append('../')

import torch
import utils
import argparse
import numpy as np
import clip
import components.data_utils as data_utils
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description='Settings for creating CBM')
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--clip_name", type=str, default="RN50", help="Which CLIP model to use")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size used when saving model/CLIP activations")
parser.add_argument("--eval_set", type=str, default="val", help="Which set to use, either 'train' or 'val'")
parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
parser.add_argument('--seed', type=int, default=0)


def get_clip_img_embedding(args):
    if args.eval_set == 'train':
        d_probe = args.dataset + '_train'
    elif args.eval_set == 'val':
        d_probe = args.dataset + '_val'
    else:
        raise ValueError('Unsupport dataset type: {}'.format(args.eval_set))
    
    # get concept set
    cls_file = data_utils.LABEL_FILES[args.dataset]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
    print(f"Dataset: {args.dataset}")
    print(f"Number of classes: {len(classes)}")

    device = torch.device(args.device)
    num_cluster = len(classes)

    clip_model, clip_preprocess = clip.load(args.clip_name, device=args.device)
    data_c = data_utils.get_data(d_probe, clip_preprocess)

    img_features = []
    labels = []
    with torch.no_grad():
        for images, label in tqdm(DataLoader(data_c, batch_size=args.batch_size, num_workers=8, pin_memory=True)):
            features = clip_model.encode_image(images.to(device))
            img_features.append(features.cpu().numpy())
            labels.append(label.cpu().numpy())

    img_features = np.vstack(img_features)
    labels = np.concatenate(labels)

    kmeans = KMeans(n_clusters=num_cluster, random_state=args.seed, n_init=1)
    predicted_labels = kmeans.fit_predict(img_features)
    w = utils.contingency_matrix(labels, predicted_labels)
    accuracy = utils.acc_from_contingency(w)
    ari = utils.adjusted_rand_index_from_contingency(w)
    nmi = utils.normalized_mutual_info_from_contingency(w)
    print()
    print(f"Clustering: ACC={accuracy}, ARI={ari}, NMI={nmi}\n")


if __name__=='__main__':
    args = parser.parse_args()
    utils.set_seed(args.seed)
    get_clip_img_embedding(args)
