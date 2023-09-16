# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 18:28:13 2021

@author: milesial

Copied from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

"""

""" Full assembly of the parts to form the complete network """
import torch
from networks.UNet_parts import *
from torch.cuda.amp import autocast

class UNetOLD(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetOLD, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        # self.dummy_param = torch.nn.Parameter(torch.empty(0))

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
    @autocast()
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x5.shape, x4.shape, x3.shape)
        # print([xx.shape for xx in [x1,x2,x3,x4,x5]])
        x = self.up1(x5, x4)
        # print(x.shape)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        # print(x.shape)
        x = self.up4(x, x1)
        # print(x.shape)
        logits = self.outc(x)
        return logits

class UNet3(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        # self.dummy_param = torch.nn.Parameter(torch.empty(0))

        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 64)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        # self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64, bilinear)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512 // factor)        
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)
        
    @autocast()
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # print(x5.shape, x4.shape, x3.shape)
        # print([xx.shape for xx in [x1,x2,x3,x4,x5]])
        x = self.up1(x4, x3)
        # print(x.shape)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        # print(x.shape)
        # x = self.up4(x, x1)
        # print(x.shape)
        logits = self.outc(x)
        return logits

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_down_layers, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_base_channels = 64
        self.n_down_layers = n_down_layers

        # https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
        # self.dummy_param = torch.nn.Parameter(torch.empty(0))
        
        self.inc = DoubleConv(n_channels, 64)
        self.down_layers = []
        # self.down_layers.append(DoubleConv(n_channels, self.n_base_channels))
        for down_layer_idx in range(n_down_layers - 1):
            self.down_layers.append(Down(self.n_base_channels*2**down_layer_idx, self.n_base_channels*2**(down_layer_idx+1)))
        # self.inc = DoubleConv(n_channels, 64)
        
        factor = 2 if bilinear else 1
        self.down_layers.append(Down(self.n_base_channels*2**(n_down_layers-1), self.n_base_channels*(2**n_down_layers) // factor))
        
        self.up_layers = []
        for up_layer_idx in range(n_down_layers - 1):
            up_layer_n_input_channels = self.n_base_channels*2**(n_down_layers - up_layer_idx)
            up_layer_n_output_channels = self.n_base_channels*2**(n_down_layers - up_layer_idx - 1) // factor
            self.up_layers.append(Up(up_layer_n_input_channels, up_layer_n_output_channels, bilinear))
        
        self.up_layers.append(
            Up(self.n_base_channels*2**(n_down_layers - (n_down_layers - 1)), self.n_base_channels, 
               bilinear))
        self.outc = OutConv(self.n_base_channels, n_classes)

        
    @autocast()
    def forward(self, x):        
        x = self.inc(x)        
        xInc = x
        down_x = []
        for down_layer_idx in range(self.n_down_layers):            
            x = self.down_layers[down_layer_idx](x)
            down_x.append(x)
        
        x = self.up_layers[0](down_x[-1], down_x[-2])
        for up_layer_idx in range(1, len(self.up_layers) - 1):
            # print(x.shape, down_x[self.n_down_layers - up_layer_idx].shape)
            x = self.up_layers[up_layer_idx](x, down_x[-up_layer_idx-2])
        x = self.up_layers[-1](x, xInc)        

        logits = self.outc(x)
        return logits



if __name__ == '__main__':
    fake_data_input = torch.rand((1,1,128,128))
    net = UNet(1, 2, 3, True)
    # net_old = UNetOLD(1, 2, True)
    # net = UNetOLD(1, 2, True)
    # fake_data_output_old = net_old(fake_data_input)
    # print('-----------')
    fake_data_output = net(fake_data_input)