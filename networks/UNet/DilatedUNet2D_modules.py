import torch
import torch.nn as nn
import torch.nn.functional as F
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, dilation=1, dropout_rate=0):
        super().__init__()        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, 
                        out_channels = out_channels, 
                        kernel_size = kernel_size, 
                        stride=1, padding=padding, dilation=dilation, groups=1, 
                        bias=True, padding_mode='reflect', 
                        device=None, dtype=None),
            nn.BatchNorm2d(out_channels, 
                            eps=1e-05, momentum=0.1, 
                            affine=True, track_running_stats=True, 
                            device=None, dtype=None),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = out_channels, 
                        out_channels = out_channels, 
                        kernel_size = kernel_size, 
                        stride=1, padding=padding, dilation=dilation, groups=1, 
                        bias=True, padding_mode='reflect', 
                        device=None, dtype=None),
            nn.BatchNorm2d(out_channels, 
                            eps=1e-05, momentum=0.1, 
                            affine=True, track_running_stats=True, 
                            device=None, dtype=None),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate, inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, dropout_rate=0):
        super().__init__()
        self.maxpool_conv_drouput = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size, padding, dilation, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv_drouput(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, bilinear=False, dropout_rate=0):
        # print('UP0!')
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        # print('UP!')
        # print(bilinear)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.bn = nn.BatchNorm2d(in_channels, 
                            eps=1e-05, momentum=0.1, 
                            affine=True, track_running_stats=True, 
                            device=None, dtype=None)
            self.conv = DoubleConv(in_channels, out_channels,kernel_size, padding='same', dropout_rate=dropout_rate)
        else:
            # print('UP!')
            # print('not bilinear!')
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                                stride = stride, 
                                padding=0, output_padding=0, groups=1, 
                                bias=True, dilation=1, 
                                device=None, dtype=None)
            self.bn = nn.BatchNorm2d(out_channels, 
                            eps=1e-05, momentum=0.1, 
                            affine=True, track_running_stats=True, 
                            device=None, dtype=None)            
            self.conv = DoubleConv(in_channels, out_channels, kernel_size, padding='same', dropout_rate=dropout_rate)

    def forward(self, x1, x2, pad = True):
        # print('x1.shape', x1.shape)
        # print('x2.shape', x2.shape)
        x1 = self.up(x1)
        # print('x1(up).shape', x1.shape)
        # print('x2.shape', x2.shape)
        x1 = self.bn(x1)
        # input is CHW
        diffY = x2.size()[-2] - x1.size()[-2]
        diffX = x2.size()[-1] - x1.size()[-1]

        # print('diffY, diffX: ', diffY, diffX)

        if pad:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # print('x1 (padded): ', x1.shape)
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = 1)

    def forward(self, x):
        return self.conv(x)