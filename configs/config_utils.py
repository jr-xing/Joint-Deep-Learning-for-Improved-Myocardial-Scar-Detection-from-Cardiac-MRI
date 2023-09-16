# https://stackoverflow.com/questions/12818146/python-argparse-ignore-unrecognised-arguments
import argparse
import copy
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    # Info
    parser.add_argument('--exp-name', dest = 'exp_name', type=str, default=argparse.SUPPRESS, help='Name of experiment')
    # Data
    # train-test-split
    # preprocessing
    parser.add_argument('--mask-out', dest = 'mask_out', type=str, default=argparse.SUPPRESS, help='Whether mask out. False or mask data type')
    parser.add_argument('--crop-to-myocardium-size', dest = 'crop_to_myocardium_size', type=str, default=argparse.SUPPRESS, help='Crop size, e.g. 120,120')
    parser.add_argument('--resize-img-size', dest = 'resize_img_size', type=str, default=argparse.SUPPRESS, help='target size, e.g. 224,224')
    # train-transform
    # validate-transform
    # network
    parser.add_argument('--load-pretrained-model', dest = 'load_pretrained_model', type=str, default=argparse.SUPPRESS, help='Whether load pretrained model')
    parser.add_argument('--load-pretrained-transformer', dest = 'load_pretrained_transformer', type=str, default=argparse.SUPPRESS, help='Whether load pretrained transformer')
    parser.add_argument('--pretrained-model-path',dest='pretrained_model_path', type=str, default=argparse.SUPPRESS, help='Path of pretrained model')
    # training
    parser.add_argument('--epochs', '-e', type=int, default=argparse.SUPPRESS, help='Number of epochs (default -1, i.e. not specified)')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=argparse.SUPPRESS, help='Batch size')
    parser.add_argument('--learning-rate', '-l', type=float, default=argparse.SUPPRESS,
                        help='Learning rate', dest='learning_rate')
    parser.add_argument('--optimizer', '-o', dest='optimizer', type=str, default=argparse.SUPPRESS, help='optimizer')
    parser.add_argument('--mixed-precision', '-amp', dest = 'amp', type=bool, default=argparse.SUPPRESS, help='Whether use mixed precision')
    parser.add_argument('--pre-load-data', dest='pre_load_data', type=bool, default=argparse.SUPPRESS, help='If load all data in memory')
    # loss
    parser.add_argument('--loss-1-weight', dest='loss_1_weight', type=float, default=argparse.SUPPRESS, help='Weight of loss 1')
    parser.add_argument('--loss-2-weight', dest='loss_2_weight', type=float, default=argparse.SUPPRESS, help='Weight of loss 2')

    # saving
    parser.add_argument('--save-nothing', dest='save_nothing', type=str, default=argparse.SUPPRESS, help='If true, save nothing')
    # others
    parser.add_argument('--config-file', dest='config_file', help='config file relative path', type=str, default='./configs/test_segmentation_config.json')
    parser.add_argument('--wandb-sweep', dest='wandb_sweep', help='whether using wandb sweep hyperparameter tuning', type=str, default='False')
    parser.add_argument('--wandb-sweep-file', dest='wandb_sweep_file', help='config file relative path', type=str, default='./configs/test_wandb_sweep.yaml')
    
    # parser.add_argument('--scale', '-s', type=float, default=argparse.SUPPRESS, help='Downscaling factor of the images')
    # parser.add_argument('--validation', '-v', dest='val', type=float, default=argparse.SUPPRESS,
    #                     help='Percent of the data that is used as validation (0-100)')
    # parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    args, undefined_args = parser.parse_known_args()
    return args, undefined_args
    # return parser.parse_args()

def update_config_by_args(config_ori, args):
    # config = config_ori.copy()
    config = copy.deepcopy(config_ori)
    for arg, value in vars(args).items():
        # Info
        if arg == 'config_file': pass
        elif arg == 'exp_name': config['info']['experiment name'] = value
        # Data
        # train-test-split
        # preprocessing
        elif arg == 'mask_out':
            if value.lower() in ['false', 'f']:
                pass
            else:
                config['preprocessing'].insert(0, {'method': 'maskout', 'mask_type': value})
        elif arg == 'crop_to_myocardium_size':
            # print("crop_to_myocardium_size!")
            # print(value)
            # print(config['preprocessing'])
            # convert value from string (e.g. '120,120') to list ([120,120])
            size = [int(val) for val in value.strip('(*)').split(',')]
            # print(size)
            # get the index of crop_to_myocardium method
            for preprocessing_dict_idx, preprocessing_dict in enumerate(config['preprocessing']):
                if preprocessing_dict['method'] == 'crop_to_myocardium':
                    preprocessing_dict['size'] = size
                    break
            print(config['preprocessing'])
        elif arg == 'resize_img_size':
            shape = [int(val) for val in value.strip('(*)').split(',')]
            preprocessing_terms = [prep['method'] for prep in config['preprocessing']]
            try:
                resize_idx = preprocessing_terms.index('resize')
                config['preprocessing']['shape'] = shape
            except:
                config['preprocessing'].insert(len(config['preprocessing']), {'method': 'resize', 'shape': shape})
        # train-transform
        # validate-transform
        # network
        elif arg == 'load_pretrained_model': config['network']['load pretrained model'] = False if value.lower() in ['false', 'f'] else True
        elif arg == 'load_pretrained_transformer': config['network']['load pretrained transformer'] = True if value.lower() in ['true', 't'] else False
        elif arg == 'pretrained_model_path': config['network']['pretrained model path'] = True if value.lower() in ['true', 't'] else False        
        # training
        elif arg == 'epochs': config['training']['epochs'] = value
        elif arg == 'batch_size': config['training']['batch size'] = value
        elif arg == 'learning_rate': config['training']['learning rate'] = value
        elif arg == 'amp': config['training']['mixed Precision'] = value
        elif arg == 'pre_load_data': config['training']['preload data'] = value
        # loss
        elif arg == 'loss_1_weight': config['loss']['input_GT_pred_role_pairs'][0]['weight'] = value
        elif arg == 'loss_2_weight': config['loss']['input_GT_pred_role_pairs'][1]['weight'] = value
        # saving
        elif arg == 'save_nothing': 
            if value.lower() in ['true', 't']:
                for k in ['save final model', 'save checkpoint', 'save prediction', 'save KeyboardInterrupt']:
                    config['saving'][k] = False
        # others        
        elif arg == 'wandb_sweep': config['others']['use wandb'] = True if value.lower() in ['true', 't', 'yes', 'y'] else False
        elif arg == 'wandb_sweep_file': config['others']['wandb sweep file'] = value        
        elif arg in ['optimizer']:
            pass
        elif arg.startswith('__'):
            pass
        else:
            raise ValueError(f'Unsupported argument: {arg}')
    
    return config

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def try_convert_to_number(s):
    try:
        int(s)
        return int(s)
    except ValueError:
        try: 
            float(s)
            return float(s)
        except ValueError:
            return s

def update_config_by_undefined_args(config_ori, undefined_args: list):
    # undefined_args should be a list of strings
    # it should follow the structure as config dict
    # the hierarchy is splitted by --
    # and the key-value is splitted by =
    # e.g. ['config--training--learning_rate=0.5'] makes config['training]['learning_rate'] = 0.5
    config = copy.deepcopy(config_ori)
    for arg_value in undefined_args:
        # print('arg_value=', arg_value)
        arg_value = arg_value.strip().lstrip('--')
        arg, value = arg_value.split('=')
        arg_split = arg.split('--')
        subconfig = config
        for layer, key in enumerate(arg_split):
            if layer < len(arg_split) - 1:
                subconfig = subconfig[key]
            else:
                subconfig[key] = try_convert_to_number(value)
    return config

import json
def load_config_from_json(json_filename=None):
    if json_filename is None:
        json_filename = './configs/test_segmentation_config.json'
    config = json.load(open(json_filename))
    return config

def update_config_by_wandb_args(
        config: dict or str, 
        wandb_metadata: dict or str,
        correct_dash = True,
        correct_underline = False):
    # base_config_file_path = './tmp/test_segmentation_config.json'
    # wandb_metadata_path = './tmp/wandb-metadata.json'
    # base_config = load_config_from_json(base_config_file_path)
    # wandb_metadata = load_config_from_json(wandb_metadata_path)    
    # Load config
    if type(config) is str:
        config = load_config_from_json(config)
    if type(wandb_metadata) is str:
        wandb_metadata = load_config_from_json(wandb_metadata)
    
    # correct args
    wandb_metadata_args = wandb_metadata['args']
    if correct_dash:
        for arg_idx, arg in enumerate(wandb_metadata_args):
            if arg.startswith('-') and not arg.startswith('--'):
                wandb_metadata_args[arg_idx] = '-'+ arg
    if correct_underline:
        for arg_idx, arg in enumerate(wandb_metadata_args):        
            wandb_metadata_args[arg_idx] = arg.replace('_', '-')
    parser = get_arg_parser()
    args, undefined_args = parser.parse_known_args(wandb_metadata['args'])
    config = update_config_by_args(config, args)
    config = update_config_by_undefined_args(config, undefined_args)

def get_arg_parser():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    # Info
    parser.add_argument('--exp-name', dest = 'exp_name', type=str, default=argparse.SUPPRESS, help='Name of experiment')
    # Data
    # train-test-split
    # preprocessing
    parser.add_argument('--mask-out', dest = 'mask_out', type=str, default=argparse.SUPPRESS, help='Whether mask out. False or mask data type')
    parser.add_argument('--crop-to-myocardium-size', dest = 'crop_to_myocardium_size', type=str, default=argparse.SUPPRESS, help='Crop size, e.g. 120,120')
    parser.add_argument('--resize-img-size', dest = 'resize_img_size', type=str, default=argparse.SUPPRESS, help='target size, e.g. 224,224')
    # train-transform
    # validate-transform
    # network
    parser.add_argument('--load-pretrained-model', dest = 'load_pretrained_model', type=str, default=argparse.SUPPRESS, help='Whether load pretrained model')
    parser.add_argument('--load-pretrained-transformer', dest = 'load_pretrained_transformer', type=str, default=argparse.SUPPRESS, help='Whether load pretrained transformer')
    parser.add_argument('--pretrained-model-path',dest='pretrained_model_path', type=str, default=argparse.SUPPRESS, help='Path of pretrained model')
    # training
    parser.add_argument('--epochs', '-e', type=int, default=argparse.SUPPRESS, help='Number of epochs (default -1, i.e. not specified)')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=argparse.SUPPRESS, help='Batch size')
    parser.add_argument('--learning-rate', '-l', type=float, default=argparse.SUPPRESS,
                        help='Learning rate', dest='learning_rate')
    parser.add_argument('--optimizer', '-o', dest='optimizer', type=str, default=argparse.SUPPRESS, help='optimizer')
    parser.add_argument('--mixed-precision', '-amp', dest = 'amp', type=bool, default=argparse.SUPPRESS, help='Whether use mixed precision')
    parser.add_argument('--pre-load-data', dest='pre_load_data', type=bool, default=argparse.SUPPRESS, help='If load all data in memory')
    # saving
    parser.add_argument('--save-nothing', dest='save_nothing', type=str, default=argparse.SUPPRESS, help='If true, save nothing')
    # others
    parser.add_argument('--config-file', dest='config_file', help='config file relative path', type=str, default='./configs/test_segmentation_config.json')
    parser.add_argument('--wandb-sweep', dest='wandb_sweep', help='whether using wandb sweep hyperparameter tuning', type=str, default='False')
    parser.add_argument('--wandb-sweep-file', dest='wandb_sweep_file', help='config file relative path', type=str, default='./configs/test_wandb_sweep.yaml')
    
    # parser.add_argument('--scale', '-s', type=float, default=argparse.SUPPRESS, help='Downscaling factor of the images')
    # parser.add_argument('--validation', '-v', dest='val', type=float, default=argparse.SUPPRESS,
    #                     help='Percent of the data that is used as validation (0-100)')
    # parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    return parser

if __name__ == '__main__':
    base_config_file_path = './tmp/test_segmentation_config.json'
    wandb_metadata_path = './tmp/wandb-metadata.json'
    base_config = load_config_from_json(base_config_file_path)
    wandb_metadata = load_config_from_json(wandb_metadata_path)
    parser = get_arg_parser()
    # correct args
    wandb_metadata_args = wandb_metadata['args']
    for arg_idx, arg in enumerate(wandb_metadata_args):
        if arg.startswith('-') and not arg.startswith('--'):
            wandb_metadata_args[arg_idx] = '-'+ arg
    for arg_idx, arg in enumerate(wandb_metadata_args):        
        wandb_metadata_args[arg_idx] = arg.replace('_', '-')


    args, undefined_args = parser.parse_known_args(wandb_metadata['args'])
    config = update_config_by_args(base_config, args)
    config = update_config_by_undefined_args(config, undefined_args)
    # for arg in wandb_metadata['args']:
    #     parser.parse_args(arg)