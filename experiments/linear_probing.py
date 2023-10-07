import os
import sys
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # make cuda deterministic
sys.path.append('../')

import torch
import torch.optim as optim
import numpy as np
import argparse
import utils
import components.data_utils as data_utils
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--dataset', type=str)
parser.add_argument('--load_hidden_concept', type=bool, default=True)
parser.add_argument('--concept_load_path', type=str, help="The path of the saved concept activation vector")
parser.add_argument('--vae_load_path', type=str, help="The path of the saved vae model")
parser.add_argument('--vae_hidden_dim', type=int, default=256)
parser.add_argument('--vae_latent_dim', type=int, default=128)
parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--save_interval', type=int, default=10)
parser.add_argument('--probe_learning_rate', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, default='cuda')

def linear_probe(args):
    device = torch.device(args.device)
    print(args)
    print()

    # get concept set
    cls_file = data_utils.LABEL_FILES[args.dataset]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
    print(f"Dataset: {args.dataset}")
    print(f"Number of classes: {len(classes)}")
    num_cluster = len(classes)

    args.load_hidden_concept = False

    if args.load_hidden_concept:
        x_train = np.load(f"{args.concept_load_path}/train_concepts.npy")
        y_train = np.load(f"{args.concept_load_path}/train_y.npy")
        x_test = np.load(f"{args.concept_load_path}/val_concepts.npy")
        y_test = np.load(f"{args.concept_load_path}/val_y.npy")

        x_train = np.reshape(x_train, (len(x_train), -1))
        x_test = np.reshape(x_test, (len(x_test), -1))
        train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    else:
        d_train = args.dataset + "_train"
        d_val = args.dataset + "_val"
        
        cbm_pipline, target_preprocess = utils.load_cbm_pipline(args.concept_load_path, device)
        train_dataset = data_utils.get_data(d_train, target_preprocess)
        test_dataset = data_utils.get_data(d_val, target_preprocess)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        train_concepts = []
        train_labels = []
        test_concepts = []
        test_labels = []

        cbm_pipline.eval()
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(train_loader):
                data = data.squeeze().to(device)
                concept_act = cbm_pipline(data)
                concept_act = concept_act.double()
                train_concepts.append(concept_act.cpu().numpy())
                train_labels.append(label.cpu().numpy())

            for batch_idx, (data, label) in enumerate(test_loader):
                data = data.squeeze().to(device)
                concept_act = cbm_pipline(data)
                concept_act = concept_act.double()
                test_concepts.append(concept_act.cpu().numpy())
                test_labels.append(label.cpu().numpy())
        
        del train_dataset, test_dataset
        del train_loader, test_loader

        train_concepts = np.concatenate(train_concepts, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        test_concepts = np.concatenate(test_concepts, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        train_concepts = torch.tensor(train_concepts, dtype=torch.double)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        test_concepts = torch.tensor(test_concepts, dtype=torch.double)
        test_labels = torch.tensor(test_labels, dtype=torch.long)


        train_dataset = TensorDataset(train_concepts, train_labels)
        test_dataset = TensorDataset(test_concepts, test_labels)

    input_dim = train_dataset[0][0].shape[0]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    vae = utils.load_vae(args.vae_load_path, input_dim, args.vae_hidden_dim, args.vae_latent_dim).double().to(device)

    probe_layer = torch.nn.Linear(in_features=args.vae_latent_dim, out_features=num_cluster, 
                                  bias=False).to(device).double()
    optimizer = optim.Adam(probe_layer.parameters(), lr=args.probe_learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    vae.eval()
    for epoch in tqdm(range(args.epochs)):
        probe_layer.train()
        total_loss = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            with torch.no_grad():
                data = data.squeeze().to(device).double()
                vae.forward(data)
            z = vae.get_z().detach()
            label = label.to(device).long()
            optimizer.zero_grad()
            logits = probe_layer(z)
            # output = F.softmax(logits, dim=1)
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
                    data = data.squeeze().to(device).double()
                    label = label.to(device).long()
                    vae.forward(data)
                    z = vae.get_z()
                    logits = probe_layer(z)
                    # output = F.softmax(logits, dim=1)
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
    classes = data_utils.get_classes(args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"Number of clusters: {len(classes)}")

    utils.set_seed(args.seed)
    linear_probe(args)
