from PIL import Image
import clip
import torch
import argparse
import warnings
import os
import torch.nn as nn
from datasets import build_dataset
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import tqdm
from torch.autograd import Variable
from metrics import *
from dbl import *

warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('MLCLIP script', add_help=False)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default='0', type=int, help='seed')
    parser.add_argument('--pretrain_clip_path', default='../pretrained/ViT-B-16.pt', type=str, help='path of pretrained clip ckpt')
    parser.add_argument('--nb_classes', default=20, type=int, help='dataset classes')
    parser.add_argument('--dataset', default='VOC', type=str, help='dataset name')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--epochs', default=30, type=int, help='train epochs')
    parser.add_argument('--from_scratch', action='store_true', default=False, help='freeze the paras of clip image encoder')
    return parser


def main(args):
    print(args)

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)

    '''
    model
    '''
    model, preprocess = clip.load(args.pretrain_clip_path, device=args.device)
    
    class visual_model(nn.Module):
        def __init__(self):
            super(visual_model,self).__init__()
            self.visual = model.visual
            self.fc = nn.Sequential(
                        nn.Linear(1024, args.nb_classes),
                        # nn.Sigmoid()
                    )
        def forward(self, x):
            x = self.visual(x).to(torch.float32)
            x = self.fc(x).to(torch.float32)
            return x

    visual_model = visual_model()
    # visual_model.load('../checkpoint/')
    visual_model.cuda()

    # freeze parameters of CLIP image encoder
    visual_model.visual.requires_grad = False
    visual_model.fc.requires_grad = True

    '''
    dataset and dataloader
    '''
    train_dataset = build_dataset(dataset=args.dataset, split='train')
    test_dataset = build_dataset(dataset=args.dataset, split='test')

    train_loader = torch.utils.data.DataLoader(
                                            train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            drop_last=False
                                        )

    test_loader = torch.utils.data.DataLoader(
                                            test_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            drop_last=False
                                        )
    '''
    loss function
    '''
    # loss_function = nn.MultiLabelSoftMarginLoss()
    # loss_function = nn.BCEWithLogitsLoss()
    if args.dataset == 'coco-lt':
        freq_file = '../data/coco/class_freq.pkl'
    elif args.dataset == 'voc-lt' or args.dataset == 'voc':
        freq_file='../data/voc/class_freq.pkl'
    if args.dataset == 'coco-lt':
        loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func='rebalance',
                focal=dict(focal=True, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=2.0, init_bias=0.05),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.2),
                loss_weight=1.0, freq_file=freq_file
            )
    elif args.dataset == 'voc-lt':
        loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func='rebalance',
                focal=dict(focal=True, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=5.0, init_bias=0.05),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
                loss_weight=1.0, freq_file=freq_file
            )

    '''
    optimizer
    '''
    if args.from_scratch is True:
        print('not freeze parameters of CLIP image encoder')
        # optimizer = optim.SGD(visual_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        optimizer = optim.AdamW(visual_model.parameters(), lr=1e-4, weight_decay=1e-5)
    else:
        print('freeze parameters of CLIP image encoder')
        # optimizer = optim.SGD(visual_model.fc.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        optimizer = optim.AdamW(visual_model.fc.parameters(), lr=1e-4, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5, mode = 'max', verbose = True, min_lr = 1e-7) 

    if args.dataset=='coco-2017' or 'coco-lt':
        dataset_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
            'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
            'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
            ]
    elif args.dataset=='voc' or 'voc-lt': 
        dataset_classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse','motorbike', 
            'person', 'pottedplant', 'sheep', 'sofa', 'train','tvmonitor'
            ]
    best_mAP = 0.0
    sf = nn.Softmax(dim=1)
    for epoch in range(args.epochs):
        visual_model.train()
        running_loss = 0.0
        gt_labels = []
        predict_p = []
        for data in tqdm.tqdm(train_loader) :
            inputs, labels = data
            labels = labels.to(torch.float32)
            labels = torch.squeeze(labels, 1)
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            optimizer.zero_grad()     
            outputs = visual_model(inputs)
            gt_labels.extend(labels.cpu().numpy().tolist())
            predict_p.extend(sf(outputs).cpu().detach().numpy())
            loss = loss_function(outputs, labels)
            running_loss += loss.data.item()
            loss.backward()
            optimizer.step()
        mAP, APs = eval_map(predict_p, gt_labels)
        print("train epoch[{}/{}] loss:{:.3f} train mAP:{}".format(epoch + 1, args.epochs, loss, mAP)) 
        visual_model.eval()
                
        with torch.no_grad():
            gt_labels = []
            predict_p = []
            running_loss = 0.0
            for data in tqdm.tqdm(test_loader):
                inputs, labels = data
                labels = labels.to(torch.float32)
                labels = torch.squeeze(labels, 1)
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                outputs = visual_model(inputs)
                gt_labels.extend(labels.cpu().numpy().tolist())
                predict_p.extend(sf(outputs).cpu().detach().numpy())
                loss = loss_function(outputs, labels)
                running_loss += loss.data.item()
            mAP, APs = eval_map(predict_p, gt_labels)
            print("test epoch[{}/{}] loss:{:.3f} test mAP:{}".format(epoch + 1, args.epochs, loss, mAP))
                
            current_mAP = mAP
            exp_lr_scheduler.step(current_mAP)
            if current_mAP > best_mAP:
                best_mAP = current_mAP
                if args.from_scratch is True:
                    torch.save(visual_model.state_dict(), f'../checkpoint/ViT16_{args.dataset}_btz_{args.batch_size}_scratch.pt') 
                    print(f'checkpoint saved at ../checkpoint/ViT16_{args.dataset}_btz_{args.batch_size}_scratch.pt')
                else:
                    torch.save(visual_model.state_dict(), f'../checkpoint/ViT16_{args.dataset}_btz_{args.batch_size}.pt') 
                    print(f'checkpoint saved at ../checkpoint/ViT16_{args.dataset}_btz_{args.batch_size}.pt')
                ltAnalysis(APs, args.dataset)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('MLCLIP script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
