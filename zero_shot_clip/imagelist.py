import os
from PIL import Image
import torch

class ImageList(object):

    def __init__(self, root, list_file, label_file):
        if isinstance(list_file, str):
            with open(list_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # ['1','2'] -> [1,2]
            def list_str2int(x: list):
                return [int(i) for i in x]

            # image path & label
            self.fns = []
            self.labels = []
            for line in lines:
                self.fns.append(line.strip().split(' ')[0])
                self.labels.append(list_str2int(line.strip().split(' ')[1:]))
            self.fns = [os.path.join(root, fn) for fn in self.fns]

            # label name
            if isinstance(label_file, str):
                with open(label_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                self.categories = [{"id": int(line.strip().split(' ')[-1]), "name": line.strip().split(' ')[0]} for line in lines]


    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        target = self.labels[idx]
        return img, target