
def get_network(net_config):
    net_name = net_config['name']
    if net_name == 'DilatedUNet':
        from networks.UNet.DilatedUNet2D import DilatedUNet2D
        net = DilatedUNet2D(n_input_channels = 1, 
            n_classes=2, 
            model_config=net_config)    
    elif net_name == 'CrossStitchingMTLDilatedUNet2D':
        from networks.UNet.MTLDilatedUNet2DDilatedUNet2D import CrossStitchingMTLDilatedUNet2D
        net = CrossStitchingMTLDilatedUNet2D(n_input_channels = 1, 
            n_classes=2, 
            model_config=net_config)
    elif net_name == 'TransUNet':
        from networks.TransUNet.vit_seg_modeling import VisionTransformer as ViT_seg
        from networks.TransUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg        
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 2
        config_vit.n_skip = 3
        config_vit.patches.grid = (14,14)
        config_vit.transformer.attention_dropout_rate = 0.3
        config_vit.transformer.dropout_rate = net_config.get('drouput', 0.3)
        net = ViT_seg(config_vit, img_size=224, num_classes=2)#.cuda()
    elif net_name == 'ComboNet':
        from networks.ComboNet.ComboNet import ComboNet
        net = ComboNet(n_input_channels = 1, 
            n_classes=2, 
            model_config=net_config)
    elif net_name == 'CrossStitchVisionTransformer':
        from networks.TransUNet.vit_seg_modeling_cross_stitch import CrossStitchVisionTransformer
        net = CrossStitchVisionTransformer(net_config)        
    elif net_name == 'HardSharingMTLVisionTransformer':
        from networks.TransUNet.vit_seg_modeling import HardSharingMTLVisionTransformer
        # from networks.TransUNet.vit_seg_modeling import VisionTransformer as ViT_seg
        from networks.TransUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg        
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 2
        config_vit.n_skip = 3
        config_vit.patches.grid = (14,14)
        config_vit.transformer.attention_dropout_rate = 0.3
        config_vit.transformer.dropout_rate = net_config.get('drouput', 0.3)
        net = HardSharingMTLVisionTransformer(config_vit, img_size=224, num_classes=2)
    return net