import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)
sys.path.append('../')

import numpy as np
import clip
import torch
from torchvision import datasets, transforms, models
from pytorchcv.model_provider import get_model as ptcv_get_model

DATASET_ROOTS = {
    "imagenet_train": "~/imagenet/ILSVRC/Data/CLS-LOC/train/",
    "imagenet_val": "~/imagenet/ILSVRC/imagenet_val",
    "cub_train":"data/CUB/train",
    "cub_val":"data/CUB/test"
}

LABEL_FILES = {"places365":"data/categories_places365_clean.txt",
               "imagenet":"data/imagenet_classes.txt",
               "cifar10":"data/cifar10_classes.txt",
               "cifar100":"data/cifar100_classes.txt",
               "cifar100-20":"data/cifar100-20_classes.txt",
               "stl10":"data/stl10_classes.txt",
               "stl10-unlabeled":"data/stl10_classes.txt",
               "cub":"data/cub_classes.txt"}

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess

def get_data(dataset_name, preprocess=None):
    if dataset_name == "cifar100_train":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                 transform=preprocess)

    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False,
                                 transform=preprocess)
        
    elif dataset_name == "cifar100-20_train":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                 transform=preprocess)
        data = get_cifar100_superclass(data)

    elif dataset_name == "cifar100-20_val":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False,
                                 transform=preprocess)
        data = get_cifar100_superclass(data)
        
    elif dataset_name == "cifar10_train":
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                transform=preprocess)
        
    elif dataset_name == "cifar10_val":
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False,
                                transform=preprocess)
        
    elif dataset_name == "stl10_train":
        data = datasets.STL10(root=os.path.expanduser("~/.cache"), download=True, split='train',
                              transform=preprocess)
        
    elif dataset_name == "stl10-unlabeled_train":
        labeled_data = datasets.STL10(root=os.path.expanduser("~/.cache"), download=True, split='train', 
                                      transform=preprocess)
        unlabeled_data = datasets.STL10(root=os.path.expanduser("~/.cache"), download=True, split='unlabeled', 
                                        transform=preprocess)
        data = torch.utils.data.ConcatDataset([unlabeled_data, labeled_data])
        data.labels = np.concatenate((unlabeled_data.labels, labeled_data.labels))
        
    elif dataset_name == "stl10_val" or dataset_name == "stl10-unlabeled_val":
        data = datasets.STL10(root=os.path.expanduser("~/.cache"), download=True, split='test', 
                              transform=preprocess)
        
    elif dataset_name == "places365_train":
        try:
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=True,
                                       transform=preprocess)
        except(RuntimeError):
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=False, 
                                      transform=preprocess)
            
    elif dataset_name == "places365_val":
        try:
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=True,
                                   transform=preprocess)
        except(RuntimeError):
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=False,
                                   transform=preprocess)
        
    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)
               
    elif dataset_name == "imagenet_broden":
        data = torch.utils.data.ConcatDataset([datasets.ImageFolder(DATASET_ROOTS["imagenet_val"], preprocess), 
                                                     datasets.ImageFolder(DATASET_ROOTS["broden"], preprocess)])
    return data

def get_targets_only(dataset_name):
    pil_data = get_data(dataset_name)
    if dataset_name.startswith("stl10"):
        return pil_data.labels
    return pil_data.targets

def get_classes(dataset):
    cls_file = LABEL_FILES[dataset]
    with open(f"../{cls_file}", "r") as f:
        classes = f.read().split("\n")
    return classes

def get_target_model(target_name, device):
    if target_name.startswith("clip_"):
        target_name = target_name[5:]
        model, preprocess = clip.load(target_name, device=device)
        target_model = lambda x: model.encode_image(x).float()
    
    elif target_name == 'resnet18_places': 
        target_model = models.resnet18(pretrained=False, num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
        
    elif target_name == 'resnet18_cub':
        target_model = ptcv_get_model("resnet18_cub", pretrained=True).to(device)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
    
    elif target_name.endswith("_v2"):
        target_name = target_name[:-3]
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V2".format(target_name_cap))
        target_model = eval("models.{}(weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
        
    else:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
    
    return target_model, preprocess

def get_cifar100_superclass(dataset):
    cifar100superclass = ["aquatic mammals","fish","flowers", "food containers", "fruit and vegetables", "household electrical devices", 
                        "household furniture", "insects", "large carnivores", "large man-made outdoor things", 
                        "large natural outdoor scenes", "large omnivores and herbivores", "medium-sized mammals",
                        "non-insect invertebrates", "people", "reptiles", "small mammals", "trees", "vehicles 1", "vehicles 2"]
    cifar100class = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        ['baby', 'boy', 'girl', 'man', 'woman'],
                        ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]
    cifar100dict = {dataset.class_to_idx[c]: i for i, sc in enumerate(cifar100superclass) for c in cifar100class[i]}
    for i, t in enumerate(dataset.targets):
        dataset.targets[i] = cifar100dict[dataset.targets[i]]
    return dataset
