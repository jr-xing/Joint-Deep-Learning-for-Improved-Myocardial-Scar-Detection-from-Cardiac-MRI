# %% 0.1 Import libraries
#!%reload_ext autoreload
#!%autoreload 2

#%%
import logging
from pathlib import Path
from threading import currentThread

import torch
import torch.nn as nn
import torchvision
import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import datetime
import numpy as np

from networks.network_utils.evaluate import evaluate_net
from networks.get_network import get_network
from utils.load_config import load_config_from_json

import json

# %% Training
if __name__ == '__main__':
# %% 1. Parse Hyper-Parameters
    from configs.config_utils import get_args, update_config_by_args, update_config_by_undefined_args
    default_seed = 4399    
    args, undefined_args = get_args()
    
    debug = False
    if debug:
        class args: 
            # config_file='./configs/default_crossstitch_myo_scar_seg_config.json'
            # config_file='./configs/default_two_step_myo_scar_seg_config.json'
            config_file='./MICCAI-2022/two-stage/myocardium-segmentation/two-stage.json'
        import os
        device_ids = [0,1]
        os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(id) for id in device_ids])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # config['training']['']
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Load configuration from file
    config = load_config_from_json(args.config_file)
    # override config file wandb setting if given command line parameter
    config = update_config_by_args(config, args)
    config = update_config_by_undefined_args(config, undefined_args)
    torch.manual_seed(config['training'].get('seed', default_seed))
    config['training']['seed'] = config['training'].get('seed', default_seed)

    net = get_network(config['network'])    
    logging.info(json.dumps(config, sort_keys=True, indent=4))

    if debug:
        # device_ids = [0,1]        
        net = nn.DataParallel(net, device_ids = device_ids)
        net.to(device=device)
    net = nn.DataParallel(net)
    net.to(device=device)
    if config['network'].get('load_pretrained_model', False):
        if config['network']['name'] == 'DilatedUNet':            
            # net.load_state_dict(torch.load(config['network']['pretrained model path'], map_location=device))
            net.load_model(config['network']['pretrained model path'], device)
            logging.info(f'Model loaded from {args.pretrained_model_path}')
        elif config['network']['name'] in ['TransUNet', 'HardSharingMTLVisionTransformer']:
            # net_preload_weights = np.load('/home/jrxing/WorkSpace/Research/Cardiac/segmentation/cardiac-segmentation/codeV2/networks/TransUNet/R50+ViT-B_16.npz')
            if config['network']['load_pretrained_transformer']:                
                # pretrained_model_path = config['network']['pretrained_model_path']
                net_preload_weights = np.load('/u/jw4hv/subspace/Research_projects/cardiac/cardiac-segmentation/codeV2/networks/TransUNet/R50+ViT-B_16.npz')
                net.module.load_from(weights=net_preload_weights)
                # net.load_transformer('/u/jw4hv/subspace/Research_projects/cardiac/cardiac-segmentation/codeV2/networks/TransUNet/R50+ViT-B_16.npz')
            else:
                net.load_model('/u/jw4hv/subspace/Research_projects/cardiac/cardiac-segmentation/codeV2/models/final-2022-01-03-14_16.pth', device)
            logging.info('pre-trained model loaded!')
    
    if config['network']['name'] in ['ComboNet', 'CrossStitchingMTLDilatedUNet2D', 'CrossStitchVisionTransformer']:
        for subnet_idx, subnet in enumerate([net.module.network1, net.module.network2]):
            subnet_role = f'network{subnet_idx+1}'
            if config['network'][subnet_role].get('load_pretrained_model', False):
                if config['network'][subnet_role]['name'] == 'DilatedUNet':
                    subnet.load_state_dict(torch.load(config[subnet_role]['pretrained model path'], map_location=device))
                elif config['network'][subnet_role]['name'] == 'TransUNet':
                    pretrained_model_path = config['network'][subnet_role]['pretrained_model_path']
                    if config['network'][subnet_role]['load_pretrained_transformer']:
                        subnet.load_from(weights=np.load(pretrained_model_path))
                    else:
                        subnet.load_state_dict(torch.load(pretrained_model_path))
                # net.load_state_dict(torch.load('/u/jw4hv/subspace/Research_projects/cardiac/cardiac-segmentation/codeV2/models/final-2022-01-03-14_16.pth', map_location=device))
                logging.info(f'Model loaded from {pretrained_model_path}')

# %% 2. Load Data
    from utils.data_io import prepare_datasets
    train_set, val_set = prepare_datasets(config)
    n_train, n_val = len(train_set), len(val_set)
    
    check_loaded = False
    if check_loaded:
        import matplotlib.pyplot as plt
        import numpy.ma as ma
        ts0 = train_set[0]
        fig, axs = plt.subplots(1,4)
        axs[0].imshow(ts0['image'][0], cmap='gray')
        axs[1].imshow(ts0['mask1'], cmap='gray')        
        axs[2].imshow(ma.array(ts0['image'][0], mask = ts0['mask1']), cmap='gray')
        axs[3].imshow(ma.array(ts0['image'][0], mask = ts0['mask2']), cmap='gray')

        import matplotlib.pyplot as plt
        import numpy.ma as ma
        td0 = train_set.data[0]
        fig, axs = plt.subplots(1,4)
        axs[0].imshow(td0['img_PSIR'], cmap='gray')
        axs[1].imshow(td0['myocardium_mask_pred_prob']>0.5, cmap='gray')
        axs[2].imshow(td0['scar_mask'], cmap='gray')        
        axs[3].imshow(ma.array(td0['img_PSIR'], mask = td0['scar_mask']), cmap='gray')

# %% 5. Create data loaders
    batch_size = config['training'].get('batch size', 20)
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
    

# %% 6. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    epochs = config['training'].get('epochs', 100)    
    learning_rate = config['training'].get('learning rate', 5e-4)    
    amp = config['training'].get('mixed Precision', True)
    use_wandb = config['others'].get('use wandb', False)
    save_checkpoint = config['training'].get('save checkpoint', True)

    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=config['training']['optimizer'].get('weight_decay', 1e-8))
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion_OLD = nn.CrossEntropyLoss()
    global_step = 0

# %% 6. Setup Experiment    
    exp_name = config['info'].get('experiment name', 'unnamed')
    exp_name_with_date = datetime.datetime.now().strftime('%Y-%m-%d')+ '-' + exp_name
    if Path(exp_name_with_date).exists():
        raise ValueError(f'Experiment {exp_name_with_date} already exists!')
    else:
        wandb_path = Path(f'./exp_results/{exp_name_with_date}-wandb')
        if use_wandb:
            wandb_path.mkdir(parents=True, exist_ok=True)    
    
    # experiment = wandb.init(
    #     project = 'trials', 
    #     name = exp_name_with_date,
    #     entity = "jrxing", 
    #     save_code = True,
    #     dir = wandb_path,
    #     resume = 'allow', 
    #     anonymous = 'must',
    #     mode='online' if use_wandb else 'disabled')
    experiment = wandb.init( 
        project = 'trials',
        entity = "jrxing", 
        save_code = True,
        dir = wandb_path,
        resume = 'allow', 
        anonymous = 'must',
        mode='online' if use_wandb else 'disabled')
    
    
    if use_wandb:
        exp_parent_path = experiment.dir
        exp_path = wandb.run.dir
    else:
        exp_parent_path = './exp_results'
        # exp_path = Path(exp_parent_path, exp_name)
        exp_path = Path(exp_parent_path, exp_name_with_date)
        exp_path.mkdir(parents=True, exist_ok=True)
        # dir_img = Path('./data/imgs/')
        # dir_mask = Path('./data/masks/')
    dir_models = Path(exp_path, 'models')
    dir_prediction = Path(exp_path, 'predictions')
    from shutil import copyfile
    copyfile(args.config_file, str(Path(exp_path, Path(args.config_file).name)))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_set)}
        Validation size: {len(val_set)}
        Checkpoints:     {save_checkpoint}
        Device:          {device}
        Mixed Precision: {amp}
    ''')

# %% 7. Begin training
    # from networks.network_utils
    from networks.network_utils.loss import CE_Dice_Loss
    criterion = CE_Dice_Loss(evaluate_role_dicts=config['loss']['input_GT_pred_role_pairs'], n_output_classes=net.module.n_classes)

    try:
        norm_img = lambda x: (x - x.min()) / (x.max()-x.min())
        # epoch = 0
        for epoch in range(epochs):
            net.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                # batch = next(iter(train_loader))
                for batch in train_loader:                    
                    for role in batch.keys():
                        batch[role] = batch[role].to(device=device)

                    with torch.cuda.amp.autocast(enabled=amp):
                        masks_pred = net(batch)
                        loss = criterion(masks_pred, batch)

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pbar.update(batch['image'].shape[0])
                    global_step += 1
                    epoch_loss += loss.item()
                    
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                # if global_step % (n_train // (10 * batch_size)) == 0:
                # net.module.encoder_cross_stitch_units[0].print_weights()
                valid_period = max(n_train // (10 * batch_size), 50)
                # if global_step % valid_period == 0:
                if epoch % valid_period == 0:
                    training_log = {
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    }
                    if use_wandb:
                        # experiment.log(training_log)
                        pass
                    else:
                        logging.info(training_log)

                    histograms = {}
                    if use_wandb:
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                    net.eval()
                    eval_paras = {
                        'method': config['evaluation']['method'],
                        'GT_type': config['evaluation']['eval_input_GT_pred_role_pairs'][0]['GT'],
                        'pred_type': config['evaluation']['eval_input_GT_pred_role_pairs'][0]['pred']
                    }
                    logging.info(f'eval_paras: {eval_paras}')
                    val_score, dice_scores, val_final_batch, val_final_batch_pred = evaluate_net(net, val_loader, device, paras=eval_paras)
                    val_loss_final_batch = criterion(val_final_batch_pred, val_final_batch)
                    net.train()
                    # scheduler.step(val_score)

                    logging.info('Validation Dice score: {}\n'.format(val_score))
                    if use_wandb:                            
                        from utils.wandb import visualize_predction
                        training_comparison = visualize_predction(batch, masks_pred, n_samples = 10,evaluate_role_dicts=config['loss']['input_GT_pred_role_pairs'])
                        valid_comparison = visualize_predction(val_final_batch, val_final_batch_pred, n_samples = 10,evaluate_role_dicts=config['loss']['input_GT_pred_role_pairs'])
                        
                        # import matplotlib.pyplot as plt
                        # plt.figure(figsize = (200,100));plt.imshow(training_comparison.moveaxis(0,-1))
                        # mask1_pred_prob = nn.Softmax(dim=1)(masks_pred['logits1']).to(torch.float32).detach().cpu()
                        # plt.imshow(mask1_pred_prob[0, 1])
                        # input2 = mask1_pred_prob[:,1:] * masks_pred['input1'].detach().cpu()
                        # plt.imshow(input2[0,0])
                        # plt.imshow(masks_pred['input2'].detach().cpu()[0,0])

                        experiment.log({
                            'training_loss': loss.item(),
                            'validation_loss': val_loss_final_batch.item(),
                            'learning_rate': optimizer.param_groups[0]['lr'],
                            'validation_Dice': val_score,
                            'valid_images': wandb.Image(valid_comparison.float()),
                            'tranining_image': wandb.Image(training_comparison.float()),
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })
                    else:
                        logging.info({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

            # if save_checkpoint:
            #     Path(dir_models).mkdir(parents=True, exist_ok=True)
            #     torch.save(net.state_dict(), str(dir_models / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            #     logging.info(f'Checkpoint {epoch + 1} saved!')        
    except KeyboardInterrupt:
        pass
    # except KeyboardInterrupt:
    #     interrupted_model_name = f'INTERRPUTED-{datetime.datetime.now().strftime("%Y-%m-%d-%H_%M")}.pth'
    #     Path(dir_models).mkdir(parents=True, exist_ok=True)
    #     torch.save(net.state_dict(), Path(dir_models, interrupted_model_name))
    #     logging.info('Saved interrupt')
        # sys.exit(0)
    
# %% 8. Prediction
    net.eval()
    final_val_loss = 0    
    from skimage.transform import rotate
    n_deploy_rotation = 9        

    # save_pred_data_GT_pred_role_pairs = \
    #     config['loss']['input_GT_pred_role_pairs'] + [{''}]
    for val_idx in range(len(val_set)):
        val_datum = val_set[val_idx]
        image_raw = val_datum['image'][0]
        
        # Initialize prediction keys
        for evaluate_role_dict in config['loss']['input_GT_pred_role_pairs']:
            val_datum[evaluate_role_dict['pred']] = 0
            val_datum[evaluate_role_dict['input']] = 0
        
        mean_mask = 0
        for rot in range(n_deploy_rotation):
            angle = rot * 360.0 / n_deploy_rotation
            image_rot = rotate(image_raw, angle, mode='edge')
            image_rot_tensor = torch.from_numpy(image_rot)[None, None, ...].to(device, dtype=torch.float32)
            masks_rot_pred = net({'image': image_rot_tensor})
            
            for evaluate_role_dict in config['loss']['input_GT_pred_role_pairs']:
                input_rot = masks_rot_pred[evaluate_role_dict['input']][0,0]
                input_rot_back = torch.from_numpy(rotate(input_rot.detach().cpu(), -angle))
                val_datum[evaluate_role_dict['input']] += input_rot_back

                mask_rot_pred = torch.moveaxis(masks_rot_pred[evaluate_role_dict['pred']].detach().cpu()[0], 0, 2)
                mask_rot_back = torch.from_numpy(rotate(mask_rot_pred, -angle))
                val_datum[evaluate_role_dict['pred']] += mask_rot_back
        # config['data']['GT_pred_role_pairs']
        for evaluate_role_dict in config['loss']['input_GT_pred_role_pairs']:
            val_datum[evaluate_role_dict['pred']] /= n_deploy_rotation
            val_datum[evaluate_role_dict['input']] /= n_deploy_rotation
        
        for key in val_datum.keys():
            if type(val_datum[key]) is torch.Tensor:
                val_datum[key] = val_datum[key].numpy()

        val_set.data[val_idx].update(val_datum)
    
    visualize_pred = False
    if visualize_pred:
        import matplotlib.pyplot as plt        
        vis_idx = 0
        vis_img = np.squeeze(val_set.data[vis_idx]['image'])
        vis_mask_GT0 = np.squeeze(val_set.data[vis_idx]['mask1'])
        vis_mask_GT1 = np.squeeze(val_set.data[vis_idx]['mask2'])
        vis_mask_pred0 = np.argmax(np.squeeze(val_set.data[vis_idx]['logits1']), axis=-1)==1
        vis_mask_pred1 = np.argmax(np.squeeze(val_set.data[vis_idx]['logits2']), axis=-1)==1
        fig, axs = plt.subplots(1,5)
        axs[0].imshow(vis_img, cmap='gray')
        axs[1].imshow(vis_mask_GT0, cmap='gray')
        axs[2].imshow(vis_mask_GT1, cmap='gray')
        axs[3].imshow(vis_mask_pred0, cmap='gray')
        axs[4].imshow(vis_mask_pred1, cmap='gray')
    
    save_model = True
    # save_best_only = True
    save_best_only = config['saving'].get('save_best_only', False)
    save_model_num = 5
    if save_model:
        current_performance = val_score.item()
        # Update Logs of all experiments
        sweep_log_dir = config['saving'].get('performance_log_dir', '/u/jw4hv/subspace/Research_projects/cardiac/cardiac-segmentation/codeV3/exp_results/sweep_logs')
        sweep_log_filename = exp_name_with_date + '.json'
        sweep_log_full_filename = str(Path(sweep_log_dir, sweep_log_filename))
        if Path(sweep_log_full_filename).is_file():
            sweep_log_dict = json.load(open(sweep_log_full_filename))
            sweep_log_dict['performance'].append(current_performance)
        else:
            sweep_log_dict = {'performance': [current_performance]}
                
        with open(sweep_log_full_filename, "w") as outfile:
            json.dump(sweep_log_dict, outfile)
        
        sweep_performances = np.array(sweep_log_dict['performance'])
        if (save_best_only \
            and len(sweep_performances) > save_model_num\
            and current_performance > np.sort(sweep_performances)[-save_model_num]) \
                or (not save_best_only)\
                or (len(sweep_performances) <= save_model_num):
    
            Path(dir_models).mkdir(parents=True, exist_ok=True)
            final_model_name = f'final-{datetime.datetime.now().strftime("%Y-%m-%d-%H_%M")}.pth'
            torch.save(net.state_dict(), Path(dir_models, final_model_name))
            logging.info(f'Model saved:'+ str(Path(dir_models, final_model_name)))

    save_prediction = True
    if save_prediction:
        from scipy import io as sio
        Path(dir_prediction).mkdir(parents=True, exist_ok=True)
        for datum_idx, datum in enumerate(val_set.data):
            filename = str(Path(
                dir_prediction,
                f"{datum['Set Name']}-{datum['Patient Name']}-{datum['Slice Name']}.mat"
            ))
            sio.savemat(filename, datum, long_field_names=True)        

    save_pred_images = True    
    if save_pred_images:
        myocardium_key = config['saving'].get('myocardium_mask_key', 'mask1')      
        import matplotlib.pyplot as plt  
        from utils.data_processing import auto_contrast_adjust
        def Dice(mask1, mask2):
            # return 2*sum(sum(mask1 * mask2)) / (sum(sum(mask1 + mask2)) - sum(sum(mask1*mask2)))
            return 2*(mask1*mask2).sum() / (mask1.sum()+mask2.sum())
        for val_idx, val_datum in enumerate(val_set.data):
            plt.ioff()
            n_rows = len(config['loss']['input_GT_pred_role_pairs'])
            n_cols = 3
            fig, axs = plt.subplots(n_rows, n_cols)
            axs = axs.flatten()
            # img = val_datum['image'].numpy()
            for GT_pred_role_pair_idx, GT_pred_role_pair in enumerate(config['loss']['input_GT_pred_role_pairs']):
                # img = val_datum[GT_pred_role_pair['input']].detach().cpu().numpy()
                img = val_datum[GT_pred_role_pair['input']]#.detach().cpu().numpy()
                mask_pred_raw = val_datum[GT_pred_role_pair['pred']]#.numpy()
                mask_pred = np.argmax(mask_pred_raw, axis=-1) == 1
                mask_gt = val_datum[GT_pred_role_pair['GT']]#.numpy()

                # https://stackoverflow.com/questions/57576686/how-to-overlay-segmented-image-on-top-of-main-image-in-python
                # Generate masked images
                xnorm = lambda img: (img - np.min(img)) / (np.max(img) - np.min(img))
                # img_normed = xnorm(img)
                img_normed = xnorm(auto_contrast_adjust(img, val_datum[myocardium_key]))
                masked_pred_img = torchvision.utils.draw_segmentation_masks(
                    image = torch.from_numpy(np.repeat(img_normed[None,...], 3, axis=0)*255).to(torch.uint8),
                    masks = torch.from_numpy(mask_pred>0.5),
                    alpha = 0.7,
                    colors = 'red'
                )
                masked_gt_img = torchvision.utils.draw_segmentation_masks(
                    image = torch.from_numpy(np.repeat(img_normed[None,...], 3, axis=0)*255).to(torch.uint8),
                    masks = torch.from_numpy(mask_gt>0.5),
                    alpha = 0.7,
                    colors = 'blue'
                )

                current_Dice = Dice(mask_pred, mask_gt)

                # Plot and save
                axs_start_idx = GT_pred_role_pair_idx * n_cols
                axs[axs_start_idx + 0].axis('off')
                axs[axs_start_idx + 0].imshow(img_normed, cmap='gray')        
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
    

    config_json_filename = Path(exp_path, 'real-config.json')
    with open(config_json_filename, "w") as outfile:
        json.dump(config, outfile)

    
    # experiment.finish()
