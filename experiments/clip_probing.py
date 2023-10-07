import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # make cuda deterministic

import torch
import torch.optim as optim
import numpy as np
import components.data_utils as data_utils
import utils
import argparse
import clip
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

parser = argparse.ArgumentParser(description='Settings for creating CBM')
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--clip_name", type=str, default="RN50", help="Which CLIP model to use")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size used when saving model/CLIP activations")
parser.add_argument('--probe_learning_rate', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--save_interval', type=int, default=10)
parser.add_argument("--device", type=str, default="cuda:1", help="Which device to use")
parser.add_argument('--seed', type=int, default=42)

def clip_probing(args):    
    cls_file = data_utils.LABEL_FILES[args.dataset]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
    print(f"Dataset: {args.dataset}")
    print(f"Number of classes: {len(classes)}")

    device = torch.device(args.device)
    num_cluster = len(classes)

    clip_model, clip_preprocess = clip.load(args.clip_name, device=args.device)
    train_dataset = data_utils.get_data(args.dataset + '_train', clip_preprocess)
    # train_label_dataset = data_utils.get_targets_only(args.dataset + '_train')
    test_dataset = data_utils.get_data(args.dataset + '_val', clip_preprocess)
    # test_label_dataset = data_utils.get_targets_only(args.dataset + '_val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    with torch.no_grad():
        for images, label in tqdm(train_loader):
            images = images.to(device, dtype=torch.float32)
            features = clip_model.encode_image(images)
            train_features.append(features.cpu().numpy())
            train_labels.append(label.cpu().numpy())

        for images, label in tqdm(test_loader):
            images = images.to(device, dtype=torch.float32)
            features = clip_model.encode_image(images)
            test_features.append(features.cpu().numpy())
            test_labels.append(label.cpu().numpy())

    del train_dataset, test_dataset
    del train_loader, test_loader

    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    test_features = np.concatenate(test_features, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    input_dim = train_features.shape[1]

    train_dataset = TensorDataset(torch.tensor(train_features, dtype=torch.float32),
                                  torch.tensor(train_labels, dtype=torch.int64))
    test_dataset = TensorDataset(torch.tensor(test_features, dtype=torch.float32),
                                 torch.tensor(test_labels, dtype=torch.int64))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    probe_layer = torch.nn.Linear(in_features=input_dim, out_features=num_cluster, 
                                  bias=False).to(device, dtype=torch.float32)
    optimizer = optim.Adam(probe_layer.parameters(), lr=args.probe_learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(args.epochs)):
        probe_layer.train()
        total_loss = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            data = data.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.int64)
            optimizer.zero_grad()
            logits = probe_layer(data)

            loss = criterion(logits, label)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        
        print(f"(Train) Epoch [{epoch+1}/{args.epochs}], Loss: {total_loss/len(train_loader.dataset):.4f}")

        if (epoch+1) % args.save_interval == 0:
            probe_layer.eval()
            test_preds = []
            test_labels = []
            total_test_loss = 0
            # correct = 0
            for batch_idx, (data, label) in enumerate(test_loader):
                with torch.no_grad():
                    data = data.to(device, dtype=torch.float32)
                    label = label.to(device, dtype=torch.int64)
                    logits = probe_layer(data)
                    loss = criterion(logits, label)
                    total_test_loss += loss.item()

                    pred = logits.argmax(dim=1, keepdim=True)
                    # correct += pred.eq(labels.view_as(pred)).sum().item()
                    test_preds.append(pred.cpu().numpy())
                    test_labels.append(label.cpu().numpy())
            test_preds = np.concatenate(test_preds, axis=0)
            test_labels = np.concatenate(test_labels, axis=0)

            w = utils.contingency_matrix(test_labels, test_preds)
            accuracy = utils.acc_from_contingency(w)
            ari = utils.adjusted_rand_index_from_contingency(w)
            nmi = utils.normalized_mutual_info_from_contingency(w)
            print(f"(Test) Epoch [{epoch+1}/{args.epochs}], Loss={total_test_loss/len(test_loader.dataset):.4f}, ACC={accuracy:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}\n")
            print()


if __name__=='__main__':
    args = parser.parse_args()
    utils.set_seed(args.seed)
    clip_probing(args)
