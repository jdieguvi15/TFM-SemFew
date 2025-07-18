import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from data.tiered_imagenet import tieredImageNet
from model.res12 import Res12
from model.swin_transformer import swin_tiny
import torch.utils.data
from utils import transform_val, transform_val_cifar, cluster
from utils import transform_val_224_cifar, transform_val_224
import open_clip
from PIL import Image

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    if args.backbone == 'resnet':
        model = Res12(avg_pool=True, drop_block='ImageNet' in args.dataset).to(device)
        model_dict = model.state_dict()
        checkpoint = torch.load(args.model_path, map_location=device)['params']
        checkpoint = {k[8:]: v for k, v in checkpoint.items()}
        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}

        print(len(checkpoint))
        model.load_state_dict(checkpoint)
        model.eval()

    elif args.backbone == 'swin':
        model = swin_tiny().to(device)
        model_dict = model.state_dict()
        checkpoint = torch.load(args.model_path, map_location=device)['params']
        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}

        print(len(checkpoint))
        model.load_state_dict(checkpoint)
        model.eval()

    elif args.backbone == 'clip':
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        model = model.to(device).eval()

    data = {}
    batch_size = 128
    shuffle = True
    
    # train
    if args.dataset == 'MiniImageNet':
        trainset = ImageFolder(args.path_to_miniimagenet + '/train', transform=transform_val if args.backbone == 'resnet' else transform_val_224 if args.backbone == 'swin' else preprocess)
    elif args.dataset == 'FC100':
        trainset = ImageFolder(args.path_to_fc100 + '/train', transform=transform_val_cifar if args.backbone == 'resnet' else transform_val_224_cifar if args.backbone == 'swin' else preprocess)
    elif args.dataset == 'CIFAR-FS':
        trainset = ImageFolder(args.path_to_cifarfs + '/train', transform=transform_val_cifar if args.backbone == 'resnet' else transform_val_224_cifar if args.backbone == 'swin' else preprocess)
    elif args.dataset == 'TieredImageNet':
        trainset = tieredImageNet(setname='train', augment=False)

        if args.backbone == 'swin':
            trainset = ImageFolder('/path/to/your/tiredimagenet/train', transform=transform_val_224)
    else:
        raise ValueError('Non-supported Dataset.')

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                              pin_memory=True)
    idx_to_class = trainset.class_to_idx
    idx_to_class = {k: v for v, k in idx_to_class.items()}
    for x, labels in tqdm(train_loader):
        labels = [idx_to_class[l.item()] for l in labels]
        with torch.no_grad():
            if args.backbone == 'clip':
                x = model.encode_image(x.to(device))
            else:
                x = model(x.to(device))
        for i, l in enumerate(labels):
            if l in data:
                data[l].append(x[i].detach().cpu().numpy())
            else:
                data[l] = [x[i].detach().cpu().numpy()]
                
    print('Finished train')

    center_mean = {}
    for k, v in data.items():
        center_mean[k] = np.array(v).mean(0)

    if args.dataset == 'TieredImageNet':
        data = {k: v[:700] for k, v in data.items()}
        center_cluster = cluster(data, len(data), 700)
    else:
        center_cluster = cluster(data, len(data), 600)

    torch.save({
        'mean': center_mean,
        'cluster': center_cluster,
        'center': center_mean
    }, 'center_{}_{}.pth'.format(args.dataset, args.backbone))

    return {'center_mean': center_mean, 'center_cluster': center_cluster, 'center': center_mean}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MiniImageNet',
                        choices=['MiniImageNet', 'TieredImageNet', 'FC100', 'CIFAR-FS'])
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--center', default='mean',
                        choices=['mean', 'cluster'])
    parser.add_argument('--backbone', default='resnet',
                        choices=['resnet', 'swin'])
                        
    parser.add_argument('--path_to_checkpoints', default='./checkpoints')
    parser.add_argument('--path_to_cifarfs', default='/Users/jdieguez/Documents/master/TFM/code/cifar-fs')
    parser.add_argument('--path_to_fc100', default='/Users/jdieguez/Documents/master/TFM/code/FC100/')
    parser.add_argument('--path_to_miniimagenet', default='/path/to/your/miniimagenet')
    parser.add_argument('--path_to_tieredimagenet', default='/path/to/your/tiredimagenet')
    
    args = parser.parse_args()
    print(vars(args))
    if args.backbone == 'resnet':
        args.model_path = f"{args.path_to_checkpoints}/ResNet-{args.dataset}.pth"
    elif args.backbone == 'swin':
        args.model_path = f"{args.path_to_checkpoints}/Swin-Tiny-{args.dataset}.pth"
    elif args.backbone == 'clip':
        args.model_path = f"{args.path_to_checkpoints}/Clip-{args.dataset}.pth"
    main(args)
