import torch
import torch.nn as nn
import torch.nn.functional as F

# class-specific loss
class SoftMarginHingeEmbeddingLoss(nn.Module):
    def __init__(self, margin=1.0, class_counts=None, gamma=2):
        super(SoftMarginHingeEmbeddingLoss, self).__init__()
        self.margin = margin
        assert self.margin > 0
        self.class_counts = class_counts
        self.gamma = gamma

    def forward(self, inputs, labels):
        # Compute the dot product between the inputs and their corresponding labels
        dot_product = torch.sum(inputs * labels, dim=2)
        margin = self.margin
        if self.class_counts is not None:
            labels = labels.long()
            self.class_counts = self.class_counts.squeeze(dim=0).expand(labels.shape[0], self.class_counts.shape[-1])
            margin = self.margin / torch.sqrt(torch.sqrt(self.class_counts.float()))
        # Compute the hinge loss
        hinge_loss = torch.relu(margin - dot_product)
        if self.class_counts is not None:
            class_weights = (1 / self.class_counts) ** self.gamma / torch.sum((1 / self.class_counts) ** self.gamma, dim=2)
            hinge_loss = hinge_loss * class_weights
            
        return hinge_loss.mean()
