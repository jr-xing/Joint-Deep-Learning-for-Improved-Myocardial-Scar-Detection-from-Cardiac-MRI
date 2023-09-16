# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 08:26:43 2021

@author: Jerry
"""

#%%
from __future__ import division
# import tensorflow as tf
# import os, re, time, glob, h5py, random
import numpy as np
from skimage import measure
import torch
from torch import nn
from networks.UNet.DilatedUNet2D_modules import *
from networks.UNet.BaseDilatedUnet2D import BaseDilatedUNet2D
#%%
class DilatedUNet2D(BaseDilatedUNet2D):
    def __init__(self, n_input_channels = 1, n_classes=2, model_config=None):
        super().__init__()
        self.n_input_channels = n_input_channels
        self.n_classes = n_classes

        self.features_root = int(model_config.get('features_root', 16))
        self.conv_size = int(model_config.get('conv_size', 3))
        self.n_layers = int(model_config.get('layers', 3))
        self.dropout_rate = float(model_config.get('dropout', 0.5))
        # self.dilations = list(map(int, model_config.get('dilations', '2,2,2,2').split(',')))
        self.dilations = int(model_config.get('dilations', 2))
        self.bilinear = model_config.get('bilinear', False)
        
        # self.inc_layers = self.build_inc_layers(
        #     n_input_channels = self.n_input_channels, 
        #     features_root = self.features_root, 
        #     conv_size = self.conv_size, 
        #     dilations = self.dilations, 
        #     dropout_rate = 0.8, 
        # )
        self.encoder = self.build_encoder(
            n_input_channels = self.n_input_channels, 
            n_layers = self.n_layers, 
            features_root = self.features_root, 
            conv_size = self.conv_size, 
            dilations = self.dilations, 
            dropout_rate = self.dropout_rate, 
            bilinear = self.bilinear)
        
        self.middle_block = self.build_middle_block(
            features_root = self.features_root, 
            n_layers = self.n_layers, 
            conv_size = self.conv_size, 
            dilations = self.dilations, 
            dropout_rate = self.dropout_rate, 
            bilinear = self.bilinear)
        
        self.decoder = self.build_decoder(
            n_layers = self.n_layers, 
            n_classes = self.n_classes, 
            features_root = self.features_root, 
            bilinear = self.bilinear, 
            dropout_rate = self.dropout_rate)
        
        self.outc = OutConv(self.features_root, self.n_classes)
    
    def forward(self, x):
        encoder_features = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            encoder_features.append(x)

        for middle_layer in self.middle_block:
            x = middle_layer(x)

        for up_layer_idx in range(0, self.n_layers):
            x = self.decoder[up_layer_idx](x, encoder_features[-up_layer_idx-1])

        logits = self.outc(x)
        # print('Final X: ', logits.shape)
        return logits    
        
if __name__ == '__main__':
    # %reload_ext autoreload
    # %autoreload 2
    fake_input = torch.rand((1,1,120,120))

    net_config= {
        'epochs': 10,
        'features_root': 64,
        'conv_size': 3,
        'im_size': '120,120',
        'layers': 5,
        'loss_type': 'MSE',
        'dropout': 0.5,
        'dilations': 1,
        'batch_size': 10,
        'deploy_rotation': 9,
        'bilinear': False
        }
    net = DilatedUNet2D(        
        n_input_channels = 1, 
        n_classes=2, 
        model_config=net_config)
    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
    #     model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # summary(model, (1, 28, 28))
    # from torchsummary import summary
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    # model = net.to(device)
    # summary(net, (1, 256, 256))


    # fake_output = net(fake_input.to(device))
    fake_output = net(fake_input)
    print(fake_output.shape)

    traced_script_module = torch.jit.trace(net, fake_input)

    # Save the TorchScript model
    traced_script_module.save("./test_UNET_torchscript.pt")
    
    