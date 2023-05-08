import torch
from torch.utils.data import Dataset
from imagelist import ImageList

class CustomDataset(Dataset):
    """Dataset.
    """

    def __init__(self, dataset, split):
        assert dataset in ["coco-lt", "voc-lt", "coco-2017", "voc"]
        if dataset == 'coco-lt':
            self.data_source = ImageList(root='../data/coco/',
                                        list_file='../data/coco/coco_lt_%s.txt' % split,
                                        label_file='../data/coco/coco_labels.txt',
                                        nb_classes=80,
                                        split=split)
        elif dataset == 'voc-lt':
            self.data_source = ImageList(root='../data/voc/',
                                        list_file='../data/voc/voc_lt_%s.txt' % split,
                                        label_file='../data/voc/voc_labels.txt',
                                        nb_classes=20,
                                        split=split)
        
        elif dataset == 'coco-2017':
            self.data_source = ImageList(root='../data/coco/',
                                        list_file='../data/coco/coco2017_%s.txt' % split,
                                        label_file='../data/coco/coco_labels.txt',
                                        nb_classes=80,
                                        split=split)
        
        elif dataset == 'voc':
            self.data_source = ImageList(root='../data/voc/',
                                        list_file='../data/voc/voc_%s.txt' % split,
                                        label_file='../data/voc/voc_labels.txt',
                                        nb_classes=20,
                                        split=split)

        self.targets = self.data_source.labels
        self.categories = self.data_source.categories
        self.fns = self.data_source.fns

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        img, target = self.data_source.get_sample(idx)
        return img, target

def build_dataset(dataset, split):
    assert split in ['train', 'test', 'val']

    assert dataset in ["coco-lt", "voc-lt", "coco-2017", "voc"]
    dataset = CustomDataset(
        dataset=dataset, 
        split=split
    )

    return dataset