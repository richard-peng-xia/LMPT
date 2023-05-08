import os
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import clip

class ImageListCaption(object):

    def __init__(self, root, list_file, caption_file, label_file, nb_classes, split, clip_model):
        with open(list_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # ['1','2'] -> [1,2]
        def list_str2int(x: list):
            return [int(i) for i in x]

        # image path & label
        self.fns = []
        self.labels = [] # one-hot e.g. [0,0,1,0,0,0,1...]
        for line in lines:
            self.fns.append(line.strip().split(' ')[0])
            label = list_str2int(line.strip().split(' ')[1:])
            # one-hot
            n_label = [[1 if i in label else 0 for i in range(nb_classes)]]
            y_all = np.ndarray([0, nb_classes])
            y_all = np.concatenate((y_all, np.array(n_label)))
            self.labels.append(y_all)
        self.labels = np.array(self.labels)
        self.fns = [os.path.join(root, fn) for fn in self.fns]

        # captions
        captions = []
        with open(caption_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            captions.append(' '.join(line.strip().split()[1:])[:75])
        self.captions_tokenized = torch.cat([clip.tokenize(c) for c in captions])
        dtype = clip_model.dtype
        with torch.no_grad():
            self.captions_embedding = clip_model.token_embedding(self.captions_tokenized).type(dtype)

        # label name
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.categories = [{"id": int(line.strip().split(' ')[-1]), "name": line.strip().split(' ')[0]} for line in lines]

        self.split = split
        self.data_transforms = {
            'train': transforms.Compose([transforms.Resize((224,224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ]),
            'test': transforms.Compose([
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
            }

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        img = self.data_transforms[self.split](img)
        target = self.labels[idx]
        caption = self.captions_tokenized[idx]
        caption_ = self.captions_embedding[idx]
        return img, target, caption, caption_

class ImageList(object):

    def __init__(self, root, list_file, label_file, nb_classes, split):
        with open(list_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # ['1','2'] -> [1,2]
        def list_str2int(x: list):
            return [int(i) for i in x]

        # image path & label
        self.fns = []
        self.labels = [] # one-hot e.g. [0,0,1,0,0,0,1...]
        for line in lines:
            self.fns.append(line.strip().split(' ')[0])
            label = list_str2int(line.strip().split(' ')[1:])
            # one-hot
            n_label = [[1 if i in label else 0 for i in range(nb_classes)]]
            y_all = np.ndarray([0, nb_classes])
            y_all = np.concatenate((y_all, np.array(n_label)))
            self.labels.append(y_all)
        self.labels = np.array(self.labels)
        self.fns = [os.path.join(root, fn) for fn in self.fns]

        # label name
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.categories = [{"id": int(line.strip().split(' ')[-1]), "name": line.strip().split(' ')[0]} for line in lines]

        self.split = split
        self.data_transforms = {
            'train': transforms.Compose([transforms.Resize((224,224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ]),
            'test': transforms.Compose([
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
            }

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        img = self.data_transforms[self.split](img)
        target = self.labels[idx]
        return img, target