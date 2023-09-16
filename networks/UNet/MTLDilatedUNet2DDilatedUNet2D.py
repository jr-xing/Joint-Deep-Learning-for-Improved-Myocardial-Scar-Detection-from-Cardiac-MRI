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
# A Cross-Stitch Architecture for Joint Registration and Segmentation in Adaptive Radiotherapy
# https://openreview.net/pdf?id=oFXY64JJQ8

#%%
class HardSharingMTLDilatedUNet2D(BaseDilatedUNet2D):
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

        self.decoder1 = self.build_decoder(
            n_layers = self.n_layers, 
            n_classes = self.n_classes, 
            features_root = self.features_root, 
            bilinear = self.bilinear, 
            dropout_rate = self.dropout_rate)
        self.outc1 = OutConv(self.features_root, self.n_classes)
        
        self.decoder2 = self.build_decoder(
            n_layers = self.n_layers, 
            n_classes = self.n_classes, 
            features_root = self.features_root, 
            bilinear = self.bilinear, 
            dropout_rate = self.dropout_rate)
        self.outc2 = OutConv(self.features_root, self.n_classes)
        
    
    def forward(self, x):
        x = x['image']
        encoder_features = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            encoder_features.append(x)

        for middle_layer in self.middle_block:
            x = middle_layer(x)

        x1 = x
        x2 = x
        for up_layer_idx in range(0, self.n_layers):
            x1 = self.decoder1[up_layer_idx](x1, encoder_features[-up_layer_idx-1])
            x2 = self.decoder2[up_layer_idx](x2, encoder_features[-up_layer_idx-1])
        
        logits1 = self.outc1(x)
        logits2 = self.outc2(x)
        # print('Final X: ', logits.shape)
        # return logits    

        # encoder_features = []
        # for layer in self.encoder:
        #     x = layer(x)
        #     encoder_features.append(x)
        
        # x1 = self.decoder1[0](encoder_features[-1], encoder_features[-2])
        # for up_layer_idx in range(1, len(self.decoder1) - 1):
        #     x1 = self.decoder1[up_layer_idx](x1, encoder_features[-up_layer_idx-2])
        # x1 = self.decoder1[-1](x1, encoder_features[0])
        # logits1 = self.outc1(x1)

        # x2 = self.decoder2[0](encoder_features[-1], encoder_features[-2])
        # for up_layer_idx in range(1, len(self.decoder1) - 1):
        #     x2 = self.decoder1[up_layer_idx](x2, encoder_features[-up_layer_idx-2])
        # x2 = self.decoder1[-1](x2, encoder_features[0])
        # logits2 = self.outc2(x2)
        return {
            'logits1': logits1,
            'logits2': logits2
        }
        # return logits1, logits2

from networks.get_network import get_network
class CrossStitchUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.w11 = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.w12 = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.w21 = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.w22 = nn.Parameter(torch.tensor(0.5, requires_grad=True))
    
    def forward(self, x1, x2):        
        x1_updated = self.w11*x1 + self.w12*x2
        x2_updated = self.w21*x1 + self.w22*x2
        return x1_updated, x2_updated
    
    def print_weights(self):
        print('weights', self.w11, self.w12, self.w21, self.w22)
        print('gradients', self.w11.grad, self.w12.grad, self.w21.grad, self.w22.grad)


class CrossStitchingMTLDilatedUNet2D(BaseDilatedUNet2D):
    def __init__(self, n_input_channels = 1, n_classes=2, model_config=None):
        super().__init__()
        self.n_input_channels = n_input_channels
        self.n_classes = n_classes

        # self.features_root = int(model_config.get('features_root', 16))
        # self.conv_size = int(model_config.get('conv_size', 3))
        # self.n_layers = int(model_config.get('layers', 3))
        # self.dropout_rate = float(model_config.get('dropout', 0.5))
        # # self.dilations = list(map(int, model_config.get('dilations', '2,2,2,2').split(',')))
        # self.dilations = int(model_config.get('dilations', 2))
        # self.bilinear = model_config.get('bilinear', False)
        # NOTE: Since cross-stitching 
        force_same_networks = model_config.get('force_same_networks', True)
        self.network1_config = model_config['network1']
        if force_same_networks:
            self.network2_config = model_config['network1']
        else:
            self.network2_config = model_config['network2']
        self.network1 = get_network(self.network1_config)
        self.network2 = get_network(self.network2_config)

        self.encoder_cross_stitch_units = nn.ModuleList()
        for down_layer_idx in range(self.network1.n_layers):
            self.encoder_cross_stitch_units.append(CrossStitchUnit())
        
        
        
        self.middle_cross_stitch_units = nn.ModuleList()
        for middle_layer_idx in range(len(self.network1.middle_block)):
            self.middle_cross_stitch_units.append(CrossStitchUnit())

        self.decoder_cross_stitch_units = nn.ModuleList()
        for up_layer_idx in range(self.network1.n_layers):
            self.decoder_cross_stitch_units.append(CrossStitchUnit())

    def forward(self, input):
        x1 = input['image']
        x2 = input['image']
        # x = input
        # The skip connection are before addition
        encoder_features1 = []
        encoder_features2 = []
        for down_layer_idx in range(self.network1.n_layers):
            x1 = self.network1.encoder[down_layer_idx](x1)
            x2 = self.network1.encoder[down_layer_idx](x2)
            encoder_features1.append(x1)
            encoder_features2.append(x2)
            # x = x1 + x2
            x1, x2 = self.encoder_cross_stitch_units[down_layer_idx](x1, x2)
                    
        for middle_layer_idx in range(len(self.network1.middle_block)):
            x1 = self.network1.middle_block[middle_layer_idx](x1)
            x2 = self.network2.middle_block[middle_layer_idx](x2)
            x1, x2 = self.middle_cross_stitch_units[middle_layer_idx](x1, x2)
            # x = x1 + x2

        for up_layer_idx in range(self.network1.n_layers):
            x1 = self.network1.decoder[up_layer_idx](x1, encoder_features1[-up_layer_idx-1])
            x2 = self.network2.decoder[up_layer_idx](x2, encoder_features2[-up_layer_idx-1])
            x1, x2 = self.decoder_cross_stitch_units[up_layer_idx](x1, x2)
            # x = x1 + x2
        
        logits1 = self.network1.outc(x1)        
        logits2 = self.network2.outc(x2)
        # return logits1, logits2
        return {
            'input1': input['image'],
            'input2': input['image'],
            'logits1': logits1,
            'logits2': logits2
        }

class CrossStitchingMTLDilatedUNet2DOLD(BaseDilatedUNet2D):
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
                
        self.encoder1 = self.build_encoder(
            n_input_channels = self.n_input_channels, 
            n_layers = self.n_layers, 
            features_root = self.features_root, 
            conv_size = self.conv_size, 
            dilations = self.dilations, 
            dropout_rate = self.dropout_rate, 
            bilinear = self.bilinear)
        self.encoder2 = self.build_encoder(
            n_input_channels = self.n_input_channels, 
            n_layers = self.n_layers, 
            features_root = self.features_root, 
            conv_size = self.conv_size, 
            dilations = self.dilations, 
            dropout_rate = self.dropout_rate, 
            bilinear = self.bilinear)

        self.middle_block1 = self.build_middle_block(
            features_root = self.features_root, 
            n_layers = self.n_layers, 
            conv_size = self.conv_size, 
            dilations = self.dilations, 
            dropout_rate = self.dropout_rate, 
            bilinear = self.bilinear)        
        self.middle_block2 = self.build_middle_block(
            features_root = self.features_root, 
            n_layers = self.n_layers, 
            conv_size = self.conv_size, 
            dilations = self.dilations, 
            dropout_rate = self.dropout_rate, 
            bilinear = self.bilinear)

        self.decoder1 = self.build_decoder(
            n_layers = self.n_layers, 
            n_classes = self.n_classes, 
            features_root = self.features_root, 
            bilinear = self.bilinear, 
            dropout_rate = self.dropout_rate)
        self.outc1 = OutConv(self.features_root, self.n_classes)
        
        self.decoder2 = self.build_decoder(
            n_layers = self.n_layers, 
            n_classes = self.n_classes, 
            features_root = self.features_root, 
            bilinear = self.bilinear, 
            dropout_rate = self.dropout_rate)
        self.outc2 = OutConv(self.features_root, self.n_classes)
        
    # def forward_alternative(self,x):
    def forward(self,x):
        x = x['image']
        # The skip connection are before addition
        encoder_features1 = []
        encoder_features2 = []
        for down_layer_idx in range(self.n_layers):
            x1 = self.encoder1[down_layer_idx](x)
            x2 = self.encoder2[down_layer_idx](x)
            encoder_features1.append(x1)
            encoder_features2.append(x2)
            x = x1 + x2
                    
        for middle_layer_idx in range(len(self.middle_block1)):
            x1 = self.middle_block1[middle_layer_idx](x)
            x2 = self.middle_block2[middle_layer_idx](x)
            x = x1 + x2

        for up_layer_idx in range(self.n_layers):
            x1 = self.decoder1[up_layer_idx](x, encoder_features1[-up_layer_idx-1])
            x2 = self.decoder2[up_layer_idx](x, encoder_features2[-up_layer_idx-1])
            x = x1 + x2
        
        logits1 = self.outc1(x)        
        logits2 = self.outc2(x)
        # return logits1, logits2
        return {
            'logits1': logits1,
            'logits2': logits2
        }
    
    def forward_alternative(self, x):
        # The skip connection are after addition
        encoder_features = []
        for down_layer_idx in range(self.n_layers):
            x1 = self.encoder1[down_layer_idx](x)
            x2 = self.encoder2[down_layer_idx](x)
            x = x1 + x2
            encoder_features.append(x)
        
        for middle_layer_idx in range(len(self.middle_block1)):
            x1 = self.middle_block1[middle_layer_idx](x)
            x2 = self.middle_block2[middle_layer_idx](x)
            x = x1 + x2

        for up_layer_idx in range(self.n_layers):
            x1 = self.decoder1[up_layer_idx](x, encoder_features[-up_layer_idx-1])
            x2 = self.decoder2[up_layer_idx](x, encoder_features[-up_layer_idx-1])
            x = x1 + x2
        
        logits1 = self.outc1(x)        
        logits2 = self.outc2(x)

        # return logits1, logits2
        return {
            'logits1': logits1,
            'logits2': logits2
        }

if __name__ == '__main__':
    # %reload_ext autoreload
    # %autoreload 2
    import torch
    from utils.load_config import load_config_from_json
    config = load_config_from_json('./configs/default_crossstitch_myo_scar_seg_config_resize.json')

    from utils.data_io import prepare_datasets
    train_set, val_set = prepare_datasets(config)

    from networks.get_network import get_network
    config['network']['network1']['layers'] = 3
    config['network']['network1']['features_root'] = 16
    config['network']['network2']['layers'] = 3
    config['network']['network2']['features_root'] = 16
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    net = get_network(config['network']).to(device=device)

    
    # from networks.UNet.MTLDilatedUNet2DDilatedUNet2D import CrossStitchingMTLDilatedUNet2D
    # fake_input = torch.rand((1,1,120,120))

    # net_config= {
    #     'epochs': 10,
    #     'features_root': 16,
    #     'conv_size': 3,
    #     'im_size': '120,120',
    #     'layers': 4,
    #     'loss_type': 'MSE',
    #     'dropout': 0.5,
    #     'dilations': 1,
    #     'batch_size': 10,
    #     'deploy_rotation': 9,
    #     'bilinear': False
    #     }
    # net_hard = CrossStitchingMTLDilatedUNet2D(        
    #     n_input_channels = 1, 
    #     n_classes=2, 
    #     model_config=net_config)

    # torch.save(net_hard, './test_net.pth')
    
    from torch.utils.data import DataLoader
    batch_size = config['training'].get('batch size', 10)
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    batch = next(iter(train_loader))

    for key in batch.keys():
        batch[key] = batch[key].to(device=device)

    batch_pred = net(batch)

    net.eval()
    final_val_loss = 0    
    from skimage.transform import rotate
    n_deploy_rotation = 9        
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    for val_idx in range(len(val_set)):
        val_datum = val_set[val_idx]
        image_raw = val_datum['image'][0]
        
        # Initialize prediction keys
        for evaluate_role_dict in config['evaluation']['input_GT_pred_role_pairs']:
            val_datum[evaluate_role_dict['pred']] = 0
            val_datum[evaluate_role_dict['input']] = 0
        
        mean_mask = 0
        for rot in range(n_deploy_rotation):
            angle = rot * 360.0 / n_deploy_rotation
            image_rot = rotate(image_raw, angle, mode='edge')
            image_rot_tensor = torch.from_numpy(image_rot)[None, None, ...].to(dtype=torch.float32)
            masks_rot_pred = net({'image': image_rot_tensor})
            
            for evaluate_role_dict in config['evaluation']['input_GT_pred_role_pairs']:
                input_rot = masks_rot_pred[evaluate_role_dict['input']][0,0]
                input_rot_back = torch.from_numpy(rotate(input_rot, -angle))
                val_datum[evaluate_role_dict['input']] += input_rot_back

                mask_rot_pred = torch.moveaxis(masks_rot_pred[evaluate_role_dict['pred']].detach().cpu()[0], 0, 2)
                mask_rot_back = torch.from_numpy(rotate(mask_rot_pred, -angle))
                val_datum[evaluate_role_dict['pred']] += mask_rot_back
        # config['data']['GT_pred_role_pairs']
        for evaluate_role_dict in config['evaluation']['input_GT_pred_role_pairs']:
            val_datum[evaluate_role_dict['pred']] /= n_deploy_rotation
            val_datum[evaluate_role_dict['input']] /= n_deploy_rotation

        val_set.data[val_idx].update(val_datum)
    
    save_pred_images = True
    if save_pred_images:      
        import torchvision
        from pathlib import Path
        import numpy as np
        dir_prediction = './tmp'
        import matplotlib.pyplot as plt  
        def Dice(mask1, mask2):
            # return 2*sum(sum(mask1 * mask2)) / (sum(sum(mask1 + mask2)) - sum(sum(mask1*mask2)))
            return 2*(mask1*mask2).sum() / (mask1.sum()+mask2.sum())
        for val_idx, val_datum in enumerate(val_set.data):
            plt.ioff()
            n_rows = len(config['evaluation']['input_GT_pred_role_pairs'])
            n_cols = 3
            fig, axs = plt.subplots(n_rows, n_cols)
            axs = axs.flatten()
            # img = val_datum['image'].numpy()
            for GT_pred_role_pair_idx, GT_pred_role_pair in enumerate(config['evaluation']['input_GT_pred_role_pairs']):
                img = val_datum[GT_pred_role_pair['input']].detach().cpu().numpy()[None,...]
                mask_pred_raw = val_datum[GT_pred_role_pair['pred']].numpy()
                mask_pred = np.argmax(mask_pred_raw, axis=-1) == 1
                mask_gt = val_datum[GT_pred_role_pair['GT']].numpy()

                # https://stackoverflow.com/questions/57576686/how-to-overlay-segmented-image-on-top-of-main-image-in-python
                # Generate masked images
                xnorm = lambda img: (img - np.min(img)) / (np.max(img) - np.min(img))
                masked_pred_img = torchvision.utils.draw_segmentation_masks(
                    image = torch.from_numpy(np.repeat(xnorm(img), 3, axis=0)*255).to(torch.uint8),
                    masks = torch.from_numpy(mask_pred>0.5),
                    alpha = 0.7,
                    colors = 'red'
                )
                masked_gt_img = torchvision.utils.draw_segmentation_masks(
                    image = torch.from_numpy(np.repeat(xnorm(img), 3, axis=0)*255).to(torch.uint8),
                    masks = torch.from_numpy(mask_gt>0.5),
                    alpha = 0.7,
                    colors = 'blue'
                )

                current_Dice = Dice(mask_pred, mask_gt)

                # Plot and save
                axs_start_idx = GT_pred_role_pair_idx * n_cols
                axs[axs_start_idx + 0].axis('off')
                axs[axs_start_idx + 0].imshow(img[0], cmap='gray')        
                axs[axs_start_idx + 0].set_title('image')
                # axs[2].imshow(ma.array(img, mask = mask_pred), cmap='gray')
                axs[axs_start_idx + 1].imshow(masked_pred_img.moveaxis(0,-1))
                axs[axs_start_idx + 1].set_title(f'Prediction\n Dice={current_Dice:.5f}')
                axs[axs_start_idx + 1].axis('off')
                axs[axs_start_idx + 2].imshow(masked_gt_img.moveaxis(0,-1))
                axs[axs_start_idx + 2].set_title('Ground Truth')
                axs[axs_start_idx + 2].axis('off')
            # axs[2].imshow(img, cmap='gray')
            # axs[2].imshow(np.ma.masked_where(mask_pred<0.5,mask_pred), cmap=mpl.cm.jet_r, alpha=0.5)
            fig.suptitle(f'{val_datum["Slice Name"]} Scar Dice = {Dice(mask_pred, mask_gt):.5f}', fontsize=16, y=1)
            
            save_fig_filename = val_datum['Set Name'] + '-' + val_datum['Patient Name'] + '-' + val_datum['Slice Name'] + '.png'
            fig.savefig(str(Path(dir_prediction, save_fig_filename)), bbox_inches='tight', dpi=250)
    # https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-convert-model
    

    #Function to Convert to ONNX 
    def Convert_ONNX(model, dummy_input, save_path): 
        # import torch.onnx 
        # set the model to inference mode 
        model.eval() 

        # Export the model   
        torch.onnx.export(model,         # model being run 
            dummy_input,       # model input (or a tuple for multiple inputs) 
            save_path,       # where to save the model  
            export_params=True,  # store the trained parameter weights inside the model file 
            opset_version=11,    # the ONNX version to export the model to 
            do_constant_folding=True,  # whether to execute constant folding for optimization 
            input_names = ['modelInput'],   # the model's input names 
            output_names = ['modelOutput'], # the model's output names 
            dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                            'modelOutput' : {0 : 'batch_size'}}) 
        print(" ") 
        print('Model has been converted to ONNX') 
    # Convert_ONNX(net_hard, fake_input, './test_ONNX.onnx')

    # Save the TorchScript model
    traced_script_module = torch.jit.trace(net, batch['image'])    
    traced_script_module.save("./tmp/test_CrossStitching_skipconnect_before_addition_torchscript.pt")