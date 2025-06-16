import torch
import torch.nn as nn
import torch.nn.functional as F
from additional_modules.arcface.backbones.model_irse import Backbone


class IDLoss(nn.Module):
    '''
    Notice that this loss assume the input face is cropped and aligned. For portrait crop,
    you need to crop and align the face first.
    '''
    def __init__(self, arcface_network_pkl):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(arcface_network_pkl, map_location='cpu'))
        self.facenet.eval()

        self.cosine_similarity = nn.CosineSimilarity()

        for p in self.facenet.parameters():
            p.requires_grad = False

    def extract_feats(self, x):
        x = F.interpolate(x, size=(112, 112), mode='bicubic')
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, x, y, reduction='mean'):
        x_feat = self.extract_feats(x)
        y_feat = self.extract_feats(y)

        icsim = -self.cosine_similarity(x_feat, y_feat)
        if reduction == 'sum':
            return icsim.sum()
        elif reduction == 'mean':
            return icsim.mean()
        return icsim
