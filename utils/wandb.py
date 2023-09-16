import torch, torchvision
def get_image_grid(images):
    # print('get_image_grid: ', images.shape)
    image_grid = torchvision.utils.make_grid(images, nrow=images.shape[0], padding=0, normalize =True, scale_each =True)
    # print('image_grid', image_grid.shape)
    return image_grid

def convert_logit_to_mask(mask_pred, target_dim = 1, append_channel_axis=True):
    # mask = torch.softmax(mask_pred, dim=1).argmax(dim=1)#.cpu().numpy()
    mask = torch.argmax(mask_pred, dim = target_dim)
    if append_channel_axis:
        mask = mask[:, None, :, :]
    return mask

def combine_image_mask(image_ori, mask, alpha = 0.5, colors = 'blue'):        
    if image_ori.max() < 1 + 1e-5:
        image = (image_ori*255).to(torch.uint8)
    else:
        image = image_ori
    # print(image.shape)
    # print(mask.shape)
    image_masked = \
        torchvision.utils.draw_segmentation_masks(
            image = image,
            masks = mask.to(torch.bool),
            alpha = alpha,
            colors = colors)
    
    return image_masked

def visualize_predction(batch, prediction, evaluate_role_dicts=None, n_samples = None):
    if evaluate_role_dicts is None:
        evaluate_role_dicts = [
            {'input': 'input1', 'pred':'logits1', 'GT':'mask1'}, 
            {'input': 'input2', 'pred':'logits2', 'GT':'mask2'}
            ]
    
    if n_samples is not None:
        n_samples = min(n_samples, batch['image'].shape[0])
    else:
        n_samples = batch['image'].shape[0]
    
    # Get grids
    # input_image_grid = get_image_grid(batch['image'][:n_samples].cpu())
    images_grid_masked = []
    for evaluate_role_dict in evaluate_role_dicts:
        # Get input
        input_image_grid = get_image_grid(prediction[evaluate_role_dict['input']][:n_samples].cpu())
        
        # Get masks        
        mask_GT_grid = get_image_grid(batch[evaluate_role_dict['GT']].cpu()[:n_samples, None, :, :].float())
        mask_pred_grid = get_image_grid(convert_logit_to_mask(prediction[evaluate_role_dict['pred']][:n_samples]).float()).cpu()
        
        # Get masked images
        image_grid_masked_GT = combine_image_mask(input_image_grid, mask_GT_grid, alpha = 0.5, colors = 'blue')
        image_grid_masked_pred = combine_image_mask(input_image_grid, mask_pred_grid, alpha = 0.5, colors = 'red')

        # Concatenate masked images
        image_grid_masked = torch.cat(
            ((input_image_grid*255).to(torch.uint8), 
            image_grid_masked_GT, 
            image_grid_masked_pred), dim=1)

        images_grid_masked.append(image_grid_masked)

    # print('input_image_grid', input_image_grid.shape)
    # print('images_grid_masked', [igm.shape for igm in images_grid_masked])
    # all_images_masked = torch.cat([input_image_grid]+images_grid_masked, dim=0)
    all_images_masked = torch.cat(images_grid_masked, dim=-1)
    return all_images_masked
    

