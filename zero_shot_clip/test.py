import clip
from PIL import Image
import torch
import argparse
import warnings
import os
from datasets import build_dataset
import time
from apmeter import *
import numpy as np
import mmcv
from mean_ap import *
from prompt_template import prompt_templates

warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('MLCLIP script', add_help=False)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
    parser.add_argument('--seed', default='0', type=int, help='seed')
    parser.add_argument('--pretrain_clip_path', default='../pretrained/RN50.pt', type=str, help='path of pretrained clip ckpt')
    parser.add_argument('--nb_classes', default=80, type=int, help='dataset classes')
    parser.add_argument('--dataset', default='COCO', type=str, help='dataset name')
    return parser

def F1_score(prob, label):
    prob = np.array(prob,dtype = bool)
    label = np.array(label,dtype = bool)
    epsilon = 1e-7
    TP = (prob & label).sum()
    TN = ((~prob) & (~label)).sum()
    FP = (prob & (~label)).sum()
    FN = ((~prob) & label).sum()
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = np.mean(TP / (TP + FP + epsilon))
    recall = np.mean(TP / (TP + FN + epsilon))
    F1scr = 2 * precision * recall / (precision + recall + epsilon)
    return precision, recall, F1scr

def ltAnalysis(mAP_cls,freq_file, dataset_classes):
    head = 0
    medium = 0
    tail = 0
    head_AP, middle_AP, tail_AP = {},{},{}
    head_count, medium_count, tail_count = 0,0,0
    class_freq = np.asarray(mmcv.load(freq_file)['class_freq'])
    hids = set(np.where(class_freq>=100)[0])
    mids = set(np.where((class_freq<100) * (class_freq >= 20))[0])
    tids = set(np.where(class_freq<20)[0])
    for id in hids:
        head_AP[dataset_classes[id]] = mAP_cls[id]
        head += mAP_cls[id]
    for id in mids:
        middle_AP[dataset_classes[id]] = mAP_cls[id]
        medium += mAP_cls[id]
    for id in tids:
        tail_AP[dataset_classes[id]] = mAP_cls[id]
        tail += mAP_cls[id]
    head = head/len(hids)
    medium = medium/len(mids)
    tail = tail/len(tids)
    print('mAP head,medium,tail:')
    print(head, medium, tail)

def main(args):
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.dataset == 'VOC':
        dataset_classes = [
                        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                        'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
                        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
                    ]
    elif args.dataset == 'COCO':
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

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)

    model, preprocess = clip.load(args.pretrain_clip_path, device=args.device, jit=False)

    dataset_test = build_dataset(dataset=args.dataset, split='test')
    
    apm_val = APMeter()

    start_time = time.time()
    zero_lists = []
    pred_all = np.ndarray([0,args.nb_classes])
    pred_raw_all = np.ndarray([0,args.nb_classes])
    y_all = np.ndarray([0,args.nb_classes])
    for i, num in enumerate(range(len(dataset_test))):
        image, label = dataset_test[num]
        if len(label)==0:
            zero_lists.append(i)
            continue
            
        categories = [dataset_test.categories[i] for i in range(args.nb_classes)]
        image_input = preprocess(image).unsqueeze(0).to(args.device)
        text = []
        for c in categories:
            text.append(clip.tokenize(' '.join([template.format(c['name']) for template in prompt_templates])[:76]))
        text_input = torch.cat(text).to(args.device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)
        n_label = [[1 if i in label else 0 for i in range(args.nb_classes)]]
        y_all = np.concatenate((y_all, np.array(n_label)))
        
        pred_raw_all = np.concatenate((pred_raw_all, similarity.cpu().detach().numpy()))
        n_pred = [[1 if i in indices else 0 for i in range(args.nb_classes)]]
        pred_all = np.concatenate((pred_all, np.array(n_pred)))
        
        if i%500==0:
            print('processing %s' %(str(i)))
        
    print("indexs of zero label:")
    print(zero_lists)
    mAP = compute_mAP(pred_raw_all, y_all)
    print("method1-mAP:", str(mAP))
    
    mAP, APs = eval_map(pred_raw_all, y_all, dataset=None, print_summary=False)
    print("method2-mAP:", str(mAP))

    if args.dataset == 'COCO':
        ltAnalysis(APs,'../data/coco/class_freq.pkl', dataset_classes=dataset_classes)
    elif args.dataset == 'VOC':
        ltAnalysis(APs,'../data/voc/class_freq.pkl', dataset_classes=dataset_classes)
    
    precision, recall , f1 = F1_score(pred_all,y_all)
    apm_val.add(torch.from_numpy(pred_raw_all), torch.from_numpy(y_all))
    mAP_val_cls = apm_val.value().cpu().detach().numpy()
    mAP_val = mAP_val_cls.mean()
    print("method3-mAP:", str(mAP_val))

    print('precision,recall,f1score:')
    print(precision,recall,f1)
        
    end_time = time.time()
    print(f'Ran for a total of {end_time-start_time} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MLCLIP script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
