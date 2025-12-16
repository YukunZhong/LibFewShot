# -*- coding: utf-8 -*-

import argparse
import torch
from torch import nn

from core.utils import accuracy
from .finetuning_model import FinetuningModel
from ..metric.deepemd import DeepEMDLayer
from torch.nn import functional as F
from core.model.backbone.resnet12_emd import custom_resnet12

class DeepEMD_Pretrain(FinetuningModel):
    def __init__(self, config, encoder_func, args, resnet12emd,**kwargs):
        super(DeepEMD_Pretrain, self).__init__()
        self.config = config
        self.train_num_class = config['train_num_class']
        self.val_num_class = config['val_num_class']
        self.feat_dim = config['feat_dim']
        self.train_classifier = nn.Linear(self.feat_dim, self.train_num_class)
        self.encoder = encoder_func
        self.args = config 
        
        self.deepemd_layer = DeepEMDLayer(self.args, mode='train', resnet12emd=encoder_func)
        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch):
            data, _ = batch
            data = self.rearrange_data(data).to(self.device)
            encoded_data = self.deepemd_layer.set_mode('encoder')(data)
            
            k = self.config['way'] * self.config['shot']
            data_shot, data_query = encoded_data[:k], encoded_data[k:]
            
            num_gpu = 1  # Assuming a fixed number of GPUs for simplicity
            logits = self.deepemd_layer.set_mode('meta')(
                (data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
            
            label = torch.arange(self.config['way'], dtype=torch.long).repeat(self.config['query']).to(self.device)
            acc = accuracy(logits, label)
            return logits, acc

    def rearrange_data(self, data):
        new_data = torch.empty_like(data)
        way, shot, query = self.config['way'], self.config['shot'], self.config['query']

        for i in range(way):
            for j in range(shot + query):
                new_idx = j * way + i if j < shot else shot * way + (j - shot) * way + i
                old_idx = i * (shot + query) + j
                new_data[new_idx] = data[old_idx]

        return new_data

    def encode(self, x, dense=True):
        x = self.encoder(x)

        if dense and self.config.get('feature_pyramid'):
            return self.build_feature_pyramid(x)
        
        return F.adaptive_avg_pool2d(x, 1).squeeze()

    def build_feature_pyramid(self, feature):

        feature_list = [F.adaptive_avg_pool2d(feature, int(size)).view(feature.shape[0], feature.shape[1], -1) for size in self.config['feature_pyramid'].split(',')]
        feature_list.append(feature.view(feature.shape[0], feature.shape[1], -1))

        return torch.cat(feature_list, dim=-1)

    def set_forward_loss(self, batch):
        image, target = batch
        image, target = image.to(self.device), target.to(self.device)

        logits = self.train_classifier(self.encode(image, dense=False))
        loss = F.cross_entropy(logits, target)
        acc = accuracy(logits, target)
        
        return logits, acc, loss


if __name__ == "__main__":
    with open('./config/deepemd_pretrain.yaml', 'r') as f:
        config = yaml.safe_load(f)

    resnet12emd = custom_resnet12()  # Initialize your ResNet12_EMD model here
    model = DeepEMD_Pretrain(config['classifier']['kwargs']['args'], resnet12emd)