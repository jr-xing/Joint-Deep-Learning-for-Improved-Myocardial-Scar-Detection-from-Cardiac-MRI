import numpy as np
import scipy.io as sio
import torch, torchvision
import matplotlib.pyplot as plt
def Dice(mask1, mask2):
    return 2*(mask1*mask2).sum() / (mask1.sum()+mask2.sum())

def get_experiment_results_from_file(
        filename, 
        img_type = 'img_PSIR',
        mask_pred_type = 'mask_pred',
        mask_gt_type = 'myocardium_pred',
        load_images = True,
        compute_Dice = True):
    exp_results = {}
    mat = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)            
    img = mat[img_type]
    mask_pred_raw = mat[mask_pred_type]
    mask_pred = np.argmax(mask_pred_raw, axis=-1) == 1
    mask_gt = mat[mask_gt_type]

    if load_images:
        exp_results['image'] = img
        exp_results['mask_pred'] = mask_pred
        exp_results['mask_gt'] = mask_gt
    
    if compute_Dice:
        dice_score = Dice(mask_pred, mask_gt)
        exp_results['Dice'] = dice_score
    
    return exp_results



    

def make_boxplots(exp_infos: list, eval_term = 'Dice'):    
    # exp_infos: list of dicts, of which each dict contrains information of the experiment
    # https://matplotlib.org/stable/gallery/pyplots/boxplot_demo_pyplot.html
    fig1, ax1 = plt.subplots()
    ax1.set_title('Basic Plot')
    ax1.boxplot(data)
    pass

if __name__ == '__main__':
    exp_results_path = '/u/jw4hv/subspace/Research_projects/cardiac/cardiac-segmentation/codeV2/exp_results/'    
    exp_infos = [
        {
            'name': '',
            'result_path': exp_results_path + r'2022-01-16-scar-seg-UNet-maskout-wandb\wandb\run-20220116_181646-dlsbbieq\files\predictions'.replace('\\', '/')
        },
        {
            'name': '',
            'result_path': exp_results_path + r'2022-01-16-scar-seg-UNet-maskout-wandb\wandb\run-20220116_181646-dlsbbieq\files\predictions'.replace('\\', '/')
        },
        {
            'name': '',
            'result_path': exp_results_path + r'2022-01-16-scar-seg-UNet-maskout-wandb\wandb\run-20220116_181646-dlsbbieq\files\predictions'.replace('\\', '/')
        },
        {
            'name': '',
            'result_path': exp_results_path + r'2022-01-16-scar-seg-UNet-maskout-wandb\wandb\run-20220116_181646-dlsbbieq\files\predictions'.replace('\\', '/')
        },
        {
            'name': '',
            'result_path': exp_results_path + r'2022-01-16-scar-seg-UNet-maskout-wandb\wandb\run-20220116_181646-dlsbbieq\files\predictions'.replace('\\', '/')
        },
        {
            'name': '',
            'result_path': exp_results_path + r'2022-01-16-scar-seg-UNet-maskout-wandb\wandb\run-20220116_181646-dlsbbieq\files\predictions'.replace('\\', '/')
        },
        {
            'name': '',
            'result_path': exp_results_path + r'2022-01-16-scar-seg-UNet-maskout-wandb\wandb\run-20220116_181646-dlsbbieq\files\predictions'.replace('\\', '/')
        }
        
    ]