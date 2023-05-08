import clip
import numpy as np
import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torch.optim as optim
import argparse
import warnings
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.autograd import Variable
from datasets import build_dataset, build_dataset_with_caption
from metrics import *
from prompt_template import prompt_templates
from coop_model import *
from asl import *
from dbl import *
# from bl import *
from csel import *

warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('auto-prompt clip', add_help=False)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default='0', type=int, help='seed')
    parser.add_argument('--pretrain_clip', default='RN50', type=str, choices=['RN50', 'ViT16'], help='pretrained clip backbone')
    parser.add_argument('--dataset', default='voc-lt', type=str, choices=['voc-lt', 'coco-lt'], help='dataset name')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=50, type=int, help='train epochs')
    parser.add_argument('--ctx_init', default='a photo of a', type=str, help='init context prompt')
    parser.add_argument('--n_ctx', default=4, type=int, help='length M of context prompt when initializing')
    parser.add_argument('--m_ctx', default=4, type=int, help='length m of context prompt for cse loss')
    parser.add_argument('--class_token_position', default='end', type=str, help='position of class token')
    parser.add_argument('--training_method', default='coop', type=str, choices=['coop', 'cocoop', 'dualcoop', 'lmpt', 'dpt'], help='training method')
    parser.add_argument('--csc', action='store_true', default=True, help='class-specific contexts (if False then initialize a generic context)')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate for optim')
    parser.add_argument('--loss_function', default='asl', type=str, choices=['asl', 'bce', 'dbl', 'mls', 'FL', 'CBloss', 'R-BCE-Focal','NTR-Focal', 'DBloss-noFocal', 'CBloss-ntr', 'DBloss'], help='loss function')
    parser.add_argument('--cseloss', default='hinge', type=str, choices=['nn_hinge', 'hinge', 'weighted', 'softmargin'], help='class-specific loss')
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam', 'adamw'], help='optimizer')
    parser.add_argument('--neg_scale', default=2.0, type=float, help='neg_scale of loss function')
    parser.add_argument('--gamma', default=2.0, type=float, help='gamma of loss function')
    parser.add_argument('--lam', default=0.5, type=float, help='lambda must be beween 0 and 1')
    return parser

def main(args):
    print(args)

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)

    """ 
    model
    """
    if args.pretrain_clip == "RN50":
        pretrain_clip_path = '../pretrained/RN50.pt'
    elif args.pretrain_clip == "ViT16":
        pretrain_clip_path = '../pretrained/ViT-B-16.pt'

    print(f"Loading CLIP (backbone: {args.pretrain_clip})")
    clip_model, preprocess = clip.load(pretrain_clip_path, device='cpu', jit=False) # Must set jit=False for training


    def convert_models_to_fp32(model): 
        for p in model.parameters(): 
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float() 
    
    clip.model.convert_weights(clip_model) # Actually this line is unnecessary since clip by default already on float16

    if args.dataset=='coco-lt':
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
    elif args.dataset=='voc-lt': 
        dataset_classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse','motorbike', 
            'person', 'pottedplant', 'sheep', 'sofa', 'train','tvmonitor'
            ]

    print("Building custom CLIP")
    model = CustomCLIP(args, classnames=dataset_classes, clip_model=clip_model)

    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad = False

    model.to(args.device)

    """
    dataset and dataloader
    """
    if args.training_method == 'lmpt':
        train_dataset = build_dataset_with_caption(dataset=args.dataset, split='train', clip_model=clip_model)
        test_dataset = build_dataset(dataset=args.dataset, split='test')
    else:
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
    """
    optimizer
    NOTE: only prompt_learner need to be updated
    """
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.prompt_learner.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
         
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.prompt_learner.parameters(), lr=args.lr, weight_decay=1e-4) 

    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.prompt_learner.parameters(), lr=args.lr, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5, mode = 'max', verbose = True, min_lr = 1e-7)
    """
    loss function
    """
    if args.dataset == 'coco-lt':
        freq_file = '../data/coco/class_freq.pkl'
    elif args.dataset == 'voc-lt':
        freq_file='../data/voc/class_freq.pkl'

    if args.loss_function == 'bce':
        loss_function = nn.BCEWithLogitsLoss()
    if args.loss_function == 'mls':
        loss_function = nn.MultiLabelSoftMarginLoss()
    if args.loss_function == 'asl':
        loss_function = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    if args.loss_function == 'dbl':
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
    
    """
    training
    """
    best_mAP = 0.0
    sf = nn.Softmax(dim=1)
    for epoch in range(args.epochs):
        model.train()
        gt_labels = []
        predict_p = []
        for data in tqdm.tqdm(train_loader):
            if args.training_method == 'lmpt':
                inputs, labels, captions, captions_ = data
                captions = Variable(captions.cuda())
                captions_ = Variable(captions_.cuda())
            else:
                inputs, labels = data
            labels = labels.to(torch.float32)
            labels = torch.squeeze(labels, 1)
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()     

            outputs = model(inputs)

            gt_labels.extend(labels.cpu().numpy().tolist())
            predict_p.extend(sf(outputs).cpu().detach().numpy())
            
            if args.training_method == 'lmpt':
                loss_1 = loss_function(outputs, labels)

                sfm = nn.Softmax(dim=0)
                class_weights = sfm(torch.from_numpy(np.asarray(mmcv.load(freq_file)['neg_class_freq'])).to(torch.float32).cuda())

                hinge_loss = SoftMarginHingeEmbeddingLoss(margin=0.2, class_counts=class_weights)

                a = captions_[:,:(77-args.m_ctx),:].unsqueeze(1).expand(captions_.shape[0], labels.shape[1], 77-args.m_ctx, captions_.shape[-1]).to(torch.float32).cuda()
                b = model.prompt_learner()[:,args.m_ctx:,:].unsqueeze(0).expand(captions_.shape[0], labels.shape[1], 77-args.m_ctx, captions_.shape[-1]).to(torch.float32).cuda()
                x = 1 - torch.cosine_similarity(a, b, dim=-1) 
                y = 2 * labels.unsqueeze(2).expand(labels.shape[0], labels.shape[1], 77-args.m_ctx) - 1

                loss_2 = hinge_loss(x, y)
                loss = args.lam * loss_1 + (1 - args.lam) * loss_2
            else:
                loss = loss_function(outputs, labels)

            loss.backward(torch.ones_like(loss))
            optimizer.step()
        try:
            mAP, APs = eval_map(predict_p, gt_labels)
            print("train epoch[{}/{}] loss:{:.3f} train mAP:{}".format(epoch + 1, args.epochs, loss, mAP)) 
        except:
            print('ValueError: Input contains NaN.')
            print("train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, args.epochs, loss)) 

        model.eval()
                
        with torch.no_grad():
            gt_labels = []
            predict_p = []
            for data in tqdm.tqdm(test_loader):
                inputs, labels = data
                labels = labels.to(torch.float32)
                labels = torch.squeeze(labels, 1)
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                outputs = model(inputs)
                gt_labels.extend(labels.cpu().numpy().tolist())
                predict_p.extend(sf(outputs).cpu().detach().numpy())
                loss = loss_function(outputs, labels)
            try:
                mAP, APs = eval_map(predict_p, gt_labels)
                print("test epoch[{}/{}] loss:{:.3f} test mAP:{}".format(epoch + 1, args.epochs, loss, mAP))
            except:
                print('ValueError: Input contains NaN.')
                print("test epoch[{}/{}] loss:{:.3f}".format(epoch + 1, args.epochs, loss))
                
            current_mAP = mAP
            exp_lr_scheduler.step(current_mAP)
            if current_mAP > best_mAP:
                best_mAP = current_mAP
                if args.csc is True:
                    if args.ctx_init == '':
                        checkpoint_path = f'../checkpoint/{args.training_method}_{args.pretrain_clip}_{args.dataset}_{args.loss_function}_{args.cseloss}_init_csc.pt'
                    else:
                        checkpoint_path = f'../checkpoint/{args.training_method}_{args.pretrain_clip}_{args.dataset}_{args.loss_function}_{args.cseloss}_csc.pt'
                else:
                    if args.ctx_init == '':
                        checkpoint_path = f'../checkpoint/{args.training_method}_{args.pretrain_clip}_{args.dataset}_{args.loss_function}_{args.cseloss}_init.pt'
                    else:
                        checkpoint_path = f'../checkpoint/{args.training_method}_{args.pretrain_clip}_{args.dataset}_{args.loss_function}_{args.cseloss}.pt'
                torch.save(model.state_dict(), checkpoint_path)
                print(f'checkpoint saved at: {checkpoint_path}')
                if args.dataset == 'coco-lt' or args.dataset == 'voc-lt':
                    ltAnalysis(APs, args.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('auto-prompt clip', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
