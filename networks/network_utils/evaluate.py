# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 10:38:50 2021

@author: Jerry
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm


# from utils.dice_score import multiclass_dice_coeff, dice_coeff
from networks.network_utils.dice_score import multiclass_dice_coeff, dice_coeff
# def binary_Dice(mask1, mask2):
#     intersection = (mask1*mask2).sum()
#     Dice_score = 2 * intersection / (mask1.sum()+mask2.sum())
#     return Dice_score


# def evaluate_batch(input, prediction, paras):
#     evaluate_method = paras.get('method', 'Dice')
#     evaluate_target = paras.get('target', 'mask2')
#     # for batch in 
#     batch_dice_score = binary_Dice(mask_pred[:, 1:, ...], prediction[evaluate_target][:, 1:, ...])

def evaluate_net(net, dataloader, device, paras={}):
    evaluate_method = paras.get('method', 'Dice')
    evaluate_GT_type = paras.get('GT_type', 'mask2')
    evaluate_pred_type = paras.get('pred_type', 'logits2')

    num_val_batches = len(dataloader)
    dice_scores = []
    dice_score = 0

    # iterate over the validation set
    for batch in dataloader:
        for role in batch.keys():
            batch[role] = batch[role].to(device=device)

        with torch.no_grad():
            # predict the mask
            prediction = net(batch)
            mask_pred = prediction[evaluate_pred_type]
            mask_true = batch[evaluate_GT_type].to(device=device)
            mask_true_onehot = F.one_hot(mask_true, net.module.n_classes).permute(0, 3, 1, 2).float()

            # convert to one-hot format
            if net.module.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                current_dice_score = dice_coeff(mask_pred, mask_true_onehot, reduce_batch_first=False)
                
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.module.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                # print(mask_pred.shape, mask_true_onehot.shape)
                current_dice_score = multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true_onehot[:, 1:, ...], reduce_batch_first=False)
                # dice_score += current_dice_score
            # mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.module.n_classes).permute(0, 3, 1, 2).float()
            # current_dice_score = binary_Dice(mask_pred[:, 1:, ...], mask_true_onehot[:, 1:, ...])

            dice_scores.append(current_dice_score)
            dice_score += current_dice_score

    return dice_score / num_val_batches, dice_scores, batch, prediction
