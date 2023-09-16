import torch.nn.functional as F
from torch import nn
from networks.network_utils.dice_score import dice_loss
class CE_Dice_Loss(nn.Module):
    def __init__(self, evaluate_role_dicts, CE_weight = 1, Dice_weight = 1, n_output_classes = 2):
        # evalutate_role_dicts should be list of role dicts to evaluate
        # e.g. evalutate_role_dicts = [{'pred':'logist1', 'GT':'mask1'}, {'pred':'logist2', 'GT':'mask2'}]
        super().__init__()
        self.evaluate_role_dicts = evaluate_role_dicts
        self.CE_loss = nn.CrossEntropyLoss()
        self.CE_weight = CE_weight
        # self.Dice_loss = dice_loss
        self.Dice_weight = Dice_weight
        self.n_output_classes = n_output_classes

        for evaluate_role_dicts in self.evaluate_role_dicts:
            if 'weight' not in evaluate_role_dicts.keys():
                evaluate_role_dicts['weight'] = 1


    def Dice_loss(self, masks_pred, true_masks):
        loss = dice_loss(F.softmax(masks_pred, dim=1).float(),
                                        F.one_hot(true_masks, self.n_output_classes).permute(0, 3, 1, 2).float(),
                                        multiclass=True)
        return loss

    def __call__(self, prediction, ground_truth):
        loss = 0
        for evaluate_role_dicts in self.evaluate_role_dicts:
            current_pred_role = evaluate_role_dicts['pred']
            current_GT_role = evaluate_role_dicts['GT']
            current_weight = evaluate_role_dicts['weight']
            loss += \
                current_weight * (
                    self.CE_loss(prediction[current_pred_role], ground_truth[current_GT_role]) * self.CE_weight + \
                    self.Dice_loss(prediction[current_pred_role], ground_truth[current_GT_role]) * self.Dice_weight
                )
        return loss