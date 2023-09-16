#%%
from threading import currentThread
import numpy as np
from pathlib import PurePath, Path

import scipy.io as sio
# from scipy.ndimage import zoom
from skimage.transform import resize
import copy
def perform_processing_seg(datum_ori, processing_list, data_types):
    # data_types: dictionary data role: data type, e.g. {image: img_PSIR, mask: myocardium_mask}
    datum = copy.deepcopy(datum_ori)
    for preprocess in processing_list:
        preprocessing_method = preprocess['method']
        if preprocessing_method in ['crop_to_myocardium']:
            # print('shape before crop:', 
                # datum[self.img_data_type].shape,
                # datum[self.mask_data_type].shape)
            box_size = preprocess.get('size', (90,90))                    
            if type(box_size) in [tuple, list]:
                if any([box_size[dim]>datum[data_types['image']].shape[dim] for dim in [-1,-2]]):
                    print(f"Error when processing data {datum['Set Name']}-{datum['Patient Name']}-{datum['Slice Name']}")
                    raise ValueError(f'Box too large. Image has shape {datum[data_types["image"]].shape} while box has shape {box_size}')
                patch_height = box_size[0]
                patch_width = box_size[1]
                box = get_fixed_size_box(
                    datum = datum, 
                    height = patch_height, 
                    width = patch_width,
                    paras = {
                        'myocardium_key':preprocess.get('myocardium_key', 'myocardium_mask'),
                        'center_type': preprocess.get('center_type', 'myocardium center')})
            else:
                raise ValueError('Unsupported box size: ', box_size, type(box_size))
            # print('box: ', box)
            crop_data_types = [data_types['image'], data_types['mask']]
            additional_data_types_to_crop = preprocess.get('additional_data_types', ['myocardium_mask'])
            crop_data_types = list(np.unique(crop_data_types + additional_data_types_to_crop))
            datum = crop_img_given_box(
                datum_ori=datum,
                box = box,
                data_types = crop_data_types,
                overlap = True)
            # print('shape after crop:', 
                # datum[self.img_data_type].shape,
                # datum[self.mask_data_type].shape)
        elif preprocessing_method in ['auto_constrast']:
            mask_type = preprocess.get('mask', 'myocardium_mask')
            datum[data_types['image']] = auto_contrast_adjust(datum[data_types['image']], datum[mask_type])
        elif preprocessing_method in ['resize']:
            if datum[data_types['image']].ndim !=2:
                raise ValueError('Image should be 2D data')
            datum[data_types['image']] = resize_image(datum[data_types['image']], preprocess['shape'], order = 3)
            datum[data_types['mask']] = resize_image(datum[data_types['mask']], preprocess['shape'], order = 0)
        elif preprocessing_method in ['normalize']:
            normlize_range = preprocess.get('range', (0,1))
            image_norm = datum[data_types['image']].astype(float)
            image_norm = (image_norm - np.min(image_norm)) / (np.max(image_norm) - np.min(image_norm))
            datum[data_types['image']] = image_norm * (normlize_range[1] - normlize_range[0]) + normlize_range[0]
        elif preprocessing_method in ['maskout']:
            mask_type = preprocess.get('mask_type', 'myocardium_mask')
            datum[data_types['image']] = combine_img_mask(datum[data_types['image']], datum[mask_type], method='maskout')
        elif preprocessing_method.startswith('skipthis-'):
            pass
        else:
            raise ValueError('Unsupported pre-processing method: ', preprocessing_method)
    return datum

def perform_processing(datum_ori, processing_list):
    # data_types: dictionary data role: data type, e.g. {image: img_PSIR, mask: myocardium_mask}
    datum = copy.deepcopy(datum_ori)
    for preprocess in processing_list:
        preprocessing_method = preprocess['method']
        preprocessing_targets = preprocess['targets']
        if preprocessing_method in ['crop_to_myocardium']:
            # print('shape before crop:', 
                # datum[self.img_data_type].shape,
                # datum[self.mask_data_type].shape)
            box_size = preprocess.get('size', (90,90))                    
            if type(box_size) in [tuple, list]:
                for preprocessing_target in preprocessing_targets:
                    if any([box_size[dim]>datum[preprocessing_target].shape[dim] for dim in [-1,-2]]):
                        print(f"Error when processing data {datum['Set Name']}-{datum['Patient Name']}-{datum['Slice Name']}")
                        raise ValueError(f'Box too large. Image has shape {datum[preprocessing_target].shape} while box has shape {box_size}')
                patch_height = box_size[0]
                patch_width = box_size[1]
                box = get_fixed_size_box(
                    datum = datum[preprocess.get('myocardium_key', 'myocardium_mask')], 
                    height = patch_height, 
                    width = patch_width,
                    paras = {
                        'center_type': preprocess.get('center_type', 'myocardium center')})
            else:
                raise ValueError('Unsupported box size: ', box_size, type(box_size))
            # print('box: ', box)
            additional_data_types_to_crop = preprocess.get('additional_data_types', ['myocardium_mask'])
            crop_data_types = list(np.unique(preprocessing_targets + additional_data_types_to_crop))
            for preprocessing_target in preprocessing_targets + additional_data_types_to_crop:
                datum[preprocessing_target] = crop_img_given_box(
                    datum=datum[preprocessing_target],
                    box = box)
            # print('shape after crop:', 
                # datum[self.img_data_type].shape,
                # datum[self.mask_data_type].shape)
        elif preprocessing_method in ['auto_constrast']:
            mask_type = preprocess.get('mask', 'myocardium_mask')
            for preprocessing_target in preprocessing_targets:
                datum[preprocessing_target] = auto_contrast_adjust(datum[preprocessing_target], datum[mask_type])
        elif preprocessing_method in ['resize']:
            # Set orders:
            if 'orders' in preprocess.keys():
                if len(preprocess['orders'])==len(preprocessing_targets):
                    orders = preprocess['orders']
                else:
                    raise ValueError('# of resize orders should be equal to # of target data types')
            else:
                orders = []
                for preprocessing_target in preprocessing_targets:
                    if any(keyword in preprocessing_target for keyword in ['mask']):
                        current_order = 0
                    else:
                        current_order = 3
                    orders.append(current_order)
            for preprocessing_target_idx, preprocessing_target in enumerate(preprocessing_targets):
                if datum[preprocessing_target].ndim !=2:
                    raise ValueError('Data to be resized should be 2D data, but has shape', datum[preprocessing_target].shape)
                datum[preprocessing_target] = resize_image(datum[preprocessing_target], preprocess['shape'], order = orders[preprocessing_target_idx])

        elif preprocessing_method in ['normalize']:
            normlize_range = preprocess.get('range', (0,1))
            for preprocessing_target in preprocessing_targets:
                image_norm = datum[preprocessing_target].astype(float)
                image_norm = (image_norm - np.min(image_norm)) / (np.max(image_norm) - np.min(image_norm))
                datum[preprocessing_target] = image_norm * (normlize_range[1] - normlize_range[0]) + normlize_range[0]
        elif preprocessing_method in ['maskout']:
            mask_type = preprocess.get('mask_type', 'myocardium_mask')
            for preprocessing_target in preprocessing_targets:
                datum[preprocessing_target] = combine_img_mask(datum[preprocessing_target], datum[mask_type], method='maskout')
        elif preprocessing_method in ['weight']:
            weight_key = preprocess.get('weight_key', 'myocardium_mask_pred_prob')
            for preprocessing_target in preprocessing_targets:
                datum[preprocessing_target] = datum[preprocessing_target]* datum[weight_key]
        elif preprocessing_method.startswith('skipthis-'):
            pass
        else:
            raise ValueError('Unsupported pre-processing method: ', preprocessing_method)
    return datum

def convert_transform_config_to_func(transform_configs):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    transform_libs = [transform['method'].split('.')[0] for transform in transform_configs]
    transform_list = []
    if all([transform_lib == 'albumentations' for transform_lib in transform_libs]):
        for transform_config in transform_configs:
            if transform_config['method'] == 'albumentations.Affine':
                transform_list.append(
                    A.Affine(
                        scale = transform_config.get('scale', (0.8, 1.2)), 
                        translate_px = transform_config.get('translate_px', (-10,10)),
                        rotate = transform_config.get('rotate', (-180, 180)), 
                        shear = transform_config.get('shear', (-5,5)), 
                        p = transform_config.get('p', 1), 
                        mode = transform_config.get('mode', 1))
                )
            elif transform_config['method'] == 'albumentations.ToTensorV2':
                transform_list.append(ToTensorV2())
        return A.Compose(transform_list)
    else:
        print('Trying to mix transform functions from libraries: ', transform_libs)
    

def resize_image(image, shape, order = 3):
    # print('image shape before zoom: ', image.shape)
    # print(image.dtype)
    image_zoomed = resize(image, shape, order=order)
    # print('image shape after zoom: ', image_zoomed.shape)
    return image_zoomed

def auto_contrast_adjust(image, mask, method=None, paras=None):
    # print(image.shape, mask.shape)
    img_mask_intensities = image[mask>0]
    image = np.minimum(
        np.maximum(image, np.min(img_mask_intensities)), 
        np.max(img_mask_intensities))
    return image

def combine_img_mask(img, mask, method='maskout'):
    # img: should be a numpy ndarray 
    # supported shapes: 
    # image: (H, W)
    # pytorch: (C, H, W), (T, C, H, W)
    if method == 'maskout':
        img_processed = img * np.squeeze(mask)
    elif method == 'fadeout':
        pass
    elif method == 'concatenate':
        pass
    return img_processed

import numpy as np
def get_myocardium_box(mask: np.ndarray):
    myocardium_y_range = np.where(np.sum(mask>0.5, axis=1)>0)
    myocardium_height = np.max(myocardium_y_range) - np.min(myocardium_y_range)
    myocardium_x_range = np.where(np.sum(mask>0.5, axis=0)>0)
    myocardium_width = np.max(myocardium_x_range) - np.min(myocardium_x_range)
    # myocardium_upper_left_location = (np.min(myocardium_y_range), np.min(myocardium_x_range))
    myocardium_center_location = [
        (np.max(myocardium_y_range) + np.min(myocardium_y_range))//2, 
        (np.max(myocardium_x_range) + np.min(myocardium_x_range))//2
    ]
    
    # return myocardium_center_location, myocardium_height, myocardium_width
    return {'center': myocardium_center_location,'height': myocardium_height, 'width': myocardium_width}

def get_joint_myocadrium_box(data: list, 
                           myocardium_mask_key: str = 'myocardium_mask',
                           power_of_2 = False,
                           even = False,
                           margin = 5):
    # 1. find the myocardium boxes
    myo_boxes = [get_myocardium_box(datum[myocardium_mask_key]) for datum in data]
    # myo_height_max = np.max([np.sum(datum[myocardium_mask_key]>0.1, axis=0) for datum in data])
    
    # 2. Determine the joint height and width
    myo_height_max = np.max([myo_box['height'] for myo_box in myo_boxes])
    myo_width_max = np.max([myo_box['width'] for myo_box in myo_boxes])
    
    if power_of_2:
        pass
    elif even:
        pass
    else:
        joint_height = myo_height_max + 2 * margin
        joint_width = myo_width_max + 2 * margin
        
    # 3. Determine new boxes
    myo_boxes_joint = []
    for datum_idx, datum in enumerate(data):
        myo_boxes_joint.append({
            'center': myo_boxes[datum_idx]['center'],
            'height': joint_height,
            'width': joint_width,
            })
        
    return myo_boxes_joint

def get_fixed_size_box_seg(datum: dict, height, width, paras):
    myocardium_key = paras.get('myocardium_key', 'myocardium_mask')
    center_type = paras.get('center_type', 'myocardium center')
    if center_type == 'myocardium center':
        myocardium_box = get_myocardium_box(datum[myocardium_key])
        myocardium_box['height'] = height
        myocardium_box['width'] = width
    elif center_type == 'image center':
        image_size = paras['image_size']
        myocardium_box = {
            'center': [image_size[0]//2, image_size[1]//2],
            'height': height,
            'width': width
            }
    elif center_type == 'given location':
        center = paras['center']
        myocardium_box = {
            'center': center,
            'height': height,
            'width': width
            }
    return myocardium_box

def get_fixed_size_box(datum: np.ndarray, height, width, paras):
    # myocardium_key = paras.get('myocardium_key', 'myocardium_mask')
    center_type = paras.get('center_type', 'myocardium center')
    if center_type == 'myocardium center':
        myocardium_box = get_myocardium_box(datum)
        myocardium_box['height'] = height
        myocardium_box['width'] = width
    elif center_type == 'image center':
        image_size = paras['image_size']
        myocardium_box = {
            'center': [image_size[0]//2, image_size[1]//2],
            'height': height,
            'width': width
            }
    elif center_type == 'given location':
        center = paras['center']
        myocardium_box = {
            'center': center,
            'height': height,
            'width': width
            }
    return myocardium_box

def crop_img_given_box_seg(datum_ori: dict, 
                           box: dict,
                           data_types: list,
                           overlap = False):
    datum = copy.deepcopy(datum_ori)
    # for datum_idx, datum in enumerate(data):
    box_center = box['center']
    box_height = box['height']
    box_width = box['width']    

    img_height = datum[data_types[0]].shape[-2]
    img_width = datum[data_types[0]].shape[-1]

    # Update box if out of boundary
    box_center[0] = np.max((box_center[0], box_height//2))
    box_center[0] = np.min((box_center[0], img_height - box_height//2))
    box_center[1] = np.max((box_center[1], box_width//2))
    box_center[1] = np.min((box_center[1], img_width - box_width//2))        

    for key in data_types:            
        if overlap:
            key_of_cropped_data = key
        else:
            key_of_cropped_data = key + '_cropped'
        datum[key_of_cropped_data] = \
            datum[key]\
                [..., box_center[0] - box_height//2: box_center[0] + box_height//2 + 0, \
                    box_center[1] - box_width//2: box_center[1] + box_width//2 + 0]
    return datum

def crop_img_given_box(datum: np.ndarray, 
                           box: dict):
    # datum = copy.deepcopy(datum_ori)
    # for datum_idx, datum in enumerate(data):
    box_center = box['center']
    box_height = box['height']
    box_width = box['width']    

    img_height = datum.shape[-2]
    img_width = datum.shape[-1]

    # Update box if out of boundary
    box_center[0] = np.max((box_center[0], box_height//2))
    box_center[0] = np.min((box_center[0], img_height - box_height//2))
    box_center[1] = np.max((box_center[1], box_width//2))
    box_center[1] = np.min((box_center[1], img_width - box_width//2))        

    return datum[..., box_center[0] - box_height//2: box_center[0] + box_height//2 + 0, \
                    box_center[1] - box_width//2: box_center[1] + box_width//2 + 0]

if __name__ == '__main__':
    from utils.data_io import load_all_filenames_from_table
    supported_suffixes = ['.ima', '.dcm', '.mat']
    filename_validation_func = lambda filename: any(suffix in filename for suffix in supported_suffixes) or '.' not in filename
    # filepath_validation_func = lambda path: 'SA' in path and ('PSIR' in path or 'MAG' in path) and 'DENSE' not in path
    filepath_validation_func = lambda path: True
    # filepath_validation_func = lambda path: 'LA' not in path and ('PSIR' in path or 'MAG' in path) and 'DENSE' not in path
    # dataset_path = str(Path('/u/jw4hv/subspace/onedrive-j-rclone/Dataset/CRT_TOS_Data_Jerry'))
    data_records_filename = str(Path('record_sheets/cardiac-LGE-dataset-2021-10-12-unet.xlsx'))
    dataset_path = str(Path('/u/jw4hv/subspace/onedrive-j-rclone/Dataset/CRT_TOS_Data_Jerry'))
    all_paths = load_all_filenames_from_table(dataset_path = dataset_path, 
                data_records_filename_relative=data_records_filename, must_has_scar=False, must_has_scar_mask=False)
    n_training = int(len(all_paths) * 4 / 5)
    training_paths = all_paths[:n_training]
    testing_paths = all_paths[n_training:]

    mat = sio.loadmat(training_paths[0], struct_as_record=False, squeeze_me=True)
    # fake_img = np.arange(3*5*5).reshape(3,5,5)
    # fake_mask = np.zeros((5, 5))
    # fake_mask[3:,3:] = 1
    # fake_img_processed = combine_img_mask(fake_img, fake_mask)