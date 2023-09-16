# import torch
from torch import nn
from networks.UNet.DilatedUNet2D_modules import *
class BaseDilatedUNet2D(nn.Module):
    def __init__(self):
        super().__init__()
    
    # def build_inc_layers(n_input_channels, features_root, conv_size, dilations, dropout_rate=0.8):
    #     inc_layers = nn.ModuleList()
    #     inc = DoubleConv(
    #         in_channels = n_input_channels, 
    #         out_channels = features_root,
    #         kernel_size = conv_size,
    #         padding = 'same',
    #         dilation = dilations,
    #         dropout_rate = dropout_rate)
    #     inc_layers.append(inc)
    #     return inc_layers
    
    def build_encoder(self, n_input_channels, n_layers, features_root, conv_size, dilations, dropout_rate, bilinear):
        # print('Created INC')
        down_layers = nn.ModuleList()
        inc = DoubleConv(
            in_channels = n_input_channels, 
            out_channels = features_root,
            kernel_size = conv_size,
            padding = 'same',
            dilation = dilations,
            dropout_rate = 0.8)
        down_layers.append(inc)
        # down_dropout_layers = nn.ModuleList()
        # down_layers.append(DoubleConv(n_channels, n_base_channels))
        for down_layer_idx in range(n_layers - 1):
            down_layers.append(Down(
                in_channels = features_root*2**down_layer_idx, 
                out_channels = features_root*2**(down_layer_idx+1),
                kernel_size = conv_size,
                padding = 'same',
                dilation = dilations,
                dropout_rate = dropout_rate
                ))
        
        # self.inc = DoubleConv(n_channels, 64)        
        # factor = 2 if bilinear else 1
        # down_layers.append(Down(
        #     in_channels = features_root*2**(n_layers-1), 
        #     out_channels = features_root*(2**n_layers) // factor,
        #     kernel_size = conv_size,
        #     padding = conv_size//2,
        #     dilation = dilations,
        #     dropout_rate = dropout_rate
        #     ))
        
        return down_layers

    def build_middle_block(self, features_root, n_layers, conv_size, dilations, dropout_rate, bilinear):
        factor = 2 if bilinear else 1
        middle_block = nn.ModuleList()
        middle_block.append(Down(
            in_channels = features_root*2**(n_layers-1), 
            out_channels = features_root*(2**n_layers) // factor,
            kernel_size = conv_size,
            padding = conv_size//2,
            dilation = dilations,
            dropout_rate = dropout_rate
            ))
        return middle_block
        
    def build_decoder(self, n_layers, n_classes, features_root, bilinear, dropout_rate=0):
        deconv_size = 2
        factor = 2 if bilinear else 1
        up_layers = nn.ModuleList()
        for up_layer_idx in range(n_layers - 1):
            up_layer_n_input_channels = features_root*2**(n_layers - up_layer_idx)
            # print(up_layer_n_input_channels)
            up_layer_n_output_channels = features_root*2**(n_layers - up_layer_idx - 1) // factor
            # print(up_layer_n_output_channels)
            # up = 
            # print('Creating UP...')
            up_layers.append(Up(
                in_channels = up_layer_n_input_channels, 
                out_channels = up_layer_n_output_channels, 
                kernel_size = deconv_size,
                bilinear = bilinear,
                dropout_rate = dropout_rate))

            # print('Created UP!')
        
        up_layers.append(
            Up(
                in_channels = features_root*2**(n_layers - (n_layers - 1)), 
                out_channels = features_root, 
                kernel_size = deconv_size,
                bilinear = bilinear,
                dropout_rate = dropout_rate))
        # print('Created UP LAYERS')
        # up_layers.append(OutConv(features_root, n_classes))
        return up_layers

    
    def load_model(self, model_filename, device):
        self.load_state_dict(torch.load(model_filename, map_location=device))