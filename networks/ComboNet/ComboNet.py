import numpy as np
from skimage import measure
import torch
from torch import nn

from networks.get_network import get_network
class ComboNet(nn.Module):
    def __init__(self, n_input_channels = 1, n_classes=2, model_config=None):
        super().__init__()
        self.network1_config = model_config['network1']
        self.network2_config = model_config['network2']
        self.network1 = get_network(self.network1_config)
        self.network2 = get_network(self.network2_config)
        self.softmax_layer = nn.Softmax(dim=1)
        self.n_classes = n_classes


    def forward(self, x):
        input1 = x['image']
        logits1 = self.network1({'image':input1})['logits']
        logits1_class1_prob = self.softmax_layer(logits1)[:,1:]
        input1_weighted = input1 * logits1_class1_prob
        logits2 = self.network2({'image':input1_weighted})['logits']

        return {
            'logits1': logits1,
            'logits2': logits2,
            'input1': input1,
            'input2': input1_weighted
        }
    
    def load_model(model_filename, target_network):
        if target_network == 'network1':
            pass
        elif target_network == 'network2':
            pass
        
if __name__ == '__main__':
    from utils.load_config import load_config_from_json
    config = load_config_from_json('./configs/default_two_step_myo_scar_seg_config.json')

    from utils.data_io import prepare_datasets
    train_set, val_set = prepare_datasets(config)

    from networks.get_network import get_network
    net = get_network(config['network'])

    from torch.utils.data import DataLoader
    batch_size = config['training'].get('batch size', 20)
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    batch = next(iter(train_loader))

    batch_pred = net(batch)