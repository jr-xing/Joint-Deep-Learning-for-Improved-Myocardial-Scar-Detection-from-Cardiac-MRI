import numpy as np
from pathlib import PurePath, Path
import pandas as pd
import SimpleITK as sitk
import scipy.io as sio
import os
import sys, platform
#%%
def get_onedrive_path():
    if sys.platform == "linux" or sys.platform == "linux2":
        # linux
        if platform.node() == 'jrxing-Tower':
            oneDrive_path = Path('/u/jw4hv/subspace/onedrive-j-rclone/')
        else:
            oneDrive_path = Path('/u/jw4hv/subspace/onedrive-j-rclone/')
        # dataset_path = Path('/u/jw4hv/subspace/onedrive-j-rclone/Dataset/CRT_TOS_Data_Jerry')
        # wandb_path = Path('/u/jw4hv/subspace/experimental_results/scar_segmentation')
    elif sys.platform == "darwin":
        # OS X
        pass
    elif sys.platform == "win32":
        # Windows...        
        if platform.node() == 'DESKTOP-RO5KFGS':
            oneDrive_path = Path('C:\\Users\\remus\\OneDrive')            
        elif platform.node() == 'DESKTOP-2OEAADG':
            oneDrive_path = Path('D:\\Documents\\OneDrive')            
    return oneDrive_path

def load_all_files(root, validation_name_func=lambda filename:True, validation_path_func=lambda path:True):
    filenames = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            # filenames.append(PurePath(path, subdirs, name))
            if validation_name_func(name) and validation_path_func(path):
                filenames.append(str(PurePath(path, name)))
    return filenames
            # (
            # 'C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry\\Pre_CRT_LBBB_with_scar\\34_CM_MR\\LGEs\\images\\SA\\PSIRS_MAG_101', 
            # [], 
            # '.MR..101.1.2010.10.25.1.1.1.1.1.ima'
            # )


def load_records_from_table(data_records_filename=None,
                         allow_all=False, 
                         must_allowed_for_myo_seg = True,
                         must_allowed_for_scar_seg = True,
                         must_has_scar=True, must_has_scar_mask=True,
                         n_read = -1,
                         to_dict = False):

    if data_records_filename is None:
        dataset_path = Path('C:\\Users\\remus\\OneDrive', 
            'Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry')
        data_records_filename = str(Path(dataset_path, 
                                        'record_sheets', 'cardiac-LGE-dataset-2021-10-12-unet.xlsx'))
    print('data_records_filename')
    print(data_records_filename)
    if str(data_records_filename).endswith('.xlsx'):
        data_records = pd.read_excel(data_records_filename, engine='openpyxl')
    else:
        data_records = pd.read_csv(data_records_filename)
    
    data_records_to_use = data_records
    if not allow_all:
        if must_allowed_for_myo_seg:
            data_records_to_use = data_records_to_use[data_records['Allow to Use (Myocardium Segmentation)'] == 1]
        if must_allowed_for_scar_seg:
            data_records_to_use = data_records_to_use[data_records['Allow to Use (Scar Segmentation)'] == 1]
        # data_records_to_use = data_records[data_records['Allow to Use'] == 1]
    # else:
    #     data_records_to_use = data_records
    
    
        
    if must_has_scar:
        data_records_to_use = data_records_to_use[data_records_to_use['Scar Exisits'] == 1]
    
    if must_has_scar_mask:
        data_records_to_use = data_records_to_use[data_records_to_use['Scar Mask Exisits'] == 1]
    
    if n_read != -1:
        data_records_to_use = data_records_to_use.iloc[:n_read]
    if to_dict:
        data_records_to_use = data_records_to_use.to_dict('records')
    return data_records_to_use

        
#     return data_filenames
def get_filenames_from_table(table, dataset_path,
        set_name_key = 'Set Name',
        patient_name_key = 'Patient Name',
        data_path_key = 'LGE Data Path under Patient Directory'):
    data_filenames = []
    for slice_idx in range(len(table)):
        slice_data_filename = str(Path(dataset_path,                                   
                                   table.iloc[slice_idx][set_name_key],
                                   table.iloc[slice_idx][patient_name_key],
                                   table.iloc[slice_idx][data_path_key]))
        data_filenames.append(slice_data_filename)
        
    return data_filenames

def loadDCMWithInfo_sitk(filename):
    reader = sitk.ImageFileReader()
    reader.SetFileName(filename)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    info = {}
    for k in reader.GetMetaDataKeys():
        v = reader.GetMetaData(k)
        info[k] = v
    data = sitk.GetArrayFromImage(sitk.ReadImage(filename))
    return data, info


def parse_train_test_split_str(train_test_split_str):
    # Eormat: method,paras1=val1|val2|...,paras2=val3|val4|...
    # Examples:
    # fixed_ratio,ratio=0.8
    # by_patient,test_patient_names=Pre_CRT_LBBB_with_scar-114_42_BC_MR|SET01-CT01
    # by_patient,test_patient_names=Pre_CRT_LBBB_with_scar-114_42_BC_MR|SET01-CT01,train_patient_names=SET01-CT02
    train_test_split_str_terms = train_test_split_str.split(',')
    method = train_test_split_str_terms[0]
    paras_dict = {}
    for train_test_split_str_term in train_test_split_str_terms[1:]:
        para_key = train_test_split_str_term.split('=')[0]
        para_val = train_test_split_str_term.split('=')[1]
        paras_dict[para_key] = para_val.split('|')
    return method, paras_dict    

def train_test_split(data: pd.DataFrame, method = 'fixed_ratio', paras = {'training_ratio': 0.8}):
    print('train_test_split')
    print('method', method)
    print('paras', paras)
    if method == 'fixed_ratio':
        if type(paras) is str:
            paras_dict = parse_train_test_split_str(paras)
            paras_dict['ratio'] = float(paras_dict['ratio'])
        else:
            paras_dict = paras
        train_patient_table, test_patient_table = train_test_split_by_fixed_ratio(data, paras_dict)
    elif method == 'by_patient':
        # paras={'test_patient_names': ['Pre_CRT_LBBB_with_scar-114_42_BC_MR']}
        if type(paras) is str:
            # paras= 'test_patient_names=Pre_CRT_LBBB_with_scar-114_42_BC_MR|SET01-CT01'
            # paras= 'test_patient_names=Pre_CRT_LBBB_with_scar-114_42_BC_MR|SET01-CT01,train_patient_names=SET01-CT02'
            paras_dict = parse_train_test_split_str(paras)
        else:
            paras_dict = paras
        train_patient_table, test_patient_table = train_test_split_by_patient(data, paras_dict)
    return train_patient_table, test_patient_table

def train_test_split_by_patient(data: pd.DataFrame, paras={'test_patient_names': ['Pre_CRT_LBBB_with_scar-114_42_BC_MR']}):
    test_patient_table = None
    train_patient_table = None

    # https://thispointer.com/python-pandas-select-rows-in-dataframe-by-conditions-on-multiple-columns/
    def get_rows_with_patient_names(table, include_patient_full_names:list):
        patients_table = None
        include_patient_set_names = [full_name.split('-')[0]  for full_name in include_patient_full_names]
        include_patient_names = [full_name.split('-')[1]  for full_name in include_patient_full_names]
        patients_table = table[(table['Set Name'].isin(include_patient_set_names)) & (table['Patient Name'].isin(include_patient_names))]
        return patients_table
    
    def get_rows_without_patient_names(table, exclude_patient_full_names:list):
        patients_table = None
        exclude_patient_set_names = [full_name.split('-')[0]  for full_name in exclude_patient_full_names]
        exclude_patient_names = [full_name.split('-')[1]  for full_name in exclude_patient_full_names]
        patients_table = table[(~table['Set Name'].isin(exclude_patient_set_names)) | (~table['Patient Name'].isin(exclude_patient_names))]
        return patients_table    

    if 'test_patient_names' in paras.keys() and 'train_patient_names' in paras.keys():        
        # test_data = data[(data['Allow to Use'] == 1) & (data['Allow to Use'] == 1)]
        # data_records_to_use = data_records[data_records['Allow to Use'] == 1]
        train_patient_table = get_rows_with_patient_names(data, paras['train_patient_names'])
        test_patient_table = get_rows_with_patient_names(data, paras['test_patient_names'])
    elif 'test_patient_names' in paras.keys():
        train_patient_table = get_rows_without_patient_names(data, paras['test_patient_names'])
        test_patient_table = get_rows_with_patient_names(data, paras['test_patient_names'])
    elif 'train_patient_names' in paras.keys():
        train_patient_table = get_rows_with_patient_names(data, paras['train_patient_names'])
        test_patient_table = get_rows_without_patient_names(data, paras['train_patient_names'])
    else:
        raise ValueError('Should have at least one of test_patient_names and train_patient_names in paras. Current paras: ', paras)
    return train_patient_table, test_patient_table

def train_test_split_by_fixed_ratio(data: pd.DataFrame, paras={'training_ratio': 0.8}):
    if 'test_patient_names' in paras.keys() and 'train_patient_names' in paras.keys():
        pass
    elif 'test_patient_names' in paras.keys():
        pass
    elif 'train_patient_names' in paras.keys():
        pass
    else:
        raise ValueError('Should have at least one of test_patient_names and train_patient_names in paras. Current paras: ', paras)
    pass

def prepare_datasets(config):
    data_source = config['data'].get('source', {'type': 'table'})
    if data_source['type'] == 'table':
        train_set, val_set = prepare_datasets_from_table(config)
    elif data_source['type'] == 'npy':
        train_set, val_set = prepare_datasets_from_npy(config)

    return train_set, val_set

def prepare_datasets_from_npy(config):
    training_data = np.load(config['data']['source']['train_set_filename'], allow_pickle=True)
    test_data = np.load(config['data']['source']['test_set_filename'], allow_pickle=True)

    from utils.dataset import DatasetSeg, DatasetMTL
    dataset_precision = np.float32
    
    # Pre-processing function
    data_preprocessing_methods = config['preprocessing']
    #%% Augmantation
    # Border types
    # https://docs.opencv.org/4.x/d2/de8/group__core__array.html
    from utils.data_processing import convert_transform_config_to_func
    train_transform = convert_transform_config_to_func(config['train-transform'])
    val_transform = convert_transform_config_to_func(config['validate-transform'])    

    # Create Dataset
    train_set = DatasetMTL(
            data = training_data,
            data_info = config['data']['inputs'], 
            transform = train_transform,
            preprocesses=data_preprocessing_methods)
    val_set = DatasetMTL(
            data = test_data,
            data_info = config['data']['inputs'], 
            transform = val_transform,
            preprocesses=data_preprocessing_methods)  
    
    train_set.preprocesse_all()
    val_set.preprocesse_all()

    return train_set, val_set

def prepare_datasets_from_table(config):
    # data_config = config['data']
    from utils.data_io import load_records_from_table, get_onedrive_path
    # import sys, platform
    from pathlib import Path
    
    # dataset_path = str(Path(get_onedrive_path(), 
    #                       'Dataset/CRT_TOS_Data_Jerry'))        
    # table_filename = str(Path(dataset_path, 'record_sheets',
    #                       'cardiac-LGE-dataset-2022-01-02-two-step-scar-segmentation-fix.xlsx'))        
    dataset_path = config['data']['source'].get('dataset_path', 
        str(Path(get_onedrive_path(), 'Dataset/CRT_TOS_Data_Jerry')))
    table_filename = config['data']['source'].get(
        'filename', str(Path(dataset_path, 'record_sheets', 'cardiac-LGE-dataset-2022-01-02-two-step-scar-segmentation-fix.xlsx')))
    
    LGE_data = load_records_from_table(
        data_records_filename=str(table_filename), 
        allow_all=False, 
        must_allowed_for_myo_seg = config['data'].get('must_allowed_for_myo_seg', True),
        must_allowed_for_scar_seg = config['data'].get('must_allowed_for_scar_seg', True),
        must_has_scar = config['data'].get('must_has_scar', True), 
        must_has_scar_mask = config['data'].get('must_has_scar_mask', True),
        # not_unsure = True,
        n_read = -1,
        to_dict = False)

    # 3. Train-valid-test split
    from utils.data_io import train_test_split
    train_test_split_config = config['train-test-split']
    training_data, test_data = train_test_split(LGE_data, train_test_split_config['method'], train_test_split_config['paras'])
        
    
    # 4. Set Dataset
    from utils.dataset import DatasetSeg, DatasetMTL
    # dataset_precision = np.float16
    dataset_precision = np.float32
    
    # Pre-processing function
    data_preprocessing_methods = config['preprocessing']
    #%% Augmantation
    # Border types
    # https://docs.opencv.org/4.x/d2/de8/group__core__array.html
    from utils.data_processing import convert_transform_config_to_func
    train_transform = convert_transform_config_to_func(config['train-transform'])
    val_transform = convert_transform_config_to_func(config['validate-transform'])    

    # Create Dataset
    train_set = DatasetMTL(
            data = training_data,
            data_info = config['data']['inputs'], 
            transform = train_transform,
            preprocesses = data_preprocessing_methods)
    val_set = DatasetMTL(
            data = test_data,
            data_info = config['data']['inputs'], 
            transform = val_transform,
            preprocesses = data_preprocessing_methods)    
    # n_train = len(train_set)
    # n_val = len(val_set)

    # Load all data if using small dataset
    if config['training'].get('preload data', False):
        train_set.load_all_data(dataset_path=dataset_path)
        val_set.load_all_data(dataset_path=dataset_path)
    
    return train_set, val_set

# %%
if __name__ =='__main__':
    # test_module_name = 'train_test_split_by_patient'
    test_module_name = 'save_prediction'
    if test_module_name == 'test_module_name':
        patients = [ ('jack', 'G1' , 34) ,
                 ('Riti', 'G1'  , 31) ,
                 ('Aadi', 'G2' , 30) ,
                 ('Sonia', 'G2', 32) ,
                 ('Lucy', 'G2'  , 33) ,
                 ('Mike', 'G3' , 35)
                  ]
        #Create a DataFrame object
        dfObj = pd.DataFrame(patients, columns = ['Patient Name' , 'Set Name', 'Sale']) 
        train_patient_table, test_patient_table = train_test_split_by_patient(
            data=dfObj,
            paras = {'test_patient_names': ['G1-Riti', 'G2-Aadi']}
        )

        table_filename = '/u/jw4hv/subspace/onedrive-j-rclone/Dataset/CRT_TOS_Data_Jerry/record_sheets/cardiac-LGE-dataset-2021-11-26-two-step-scar-segmentation.xlsx'
        table = load_records_from_table(data_records_filename=table_filename)
        # train_test_split_method, train_test_split_paras = parse_train_test_split_str('by_patient,test_patient_names=Pre_CRT_LBBB_with_scar-114_42_BC_MR|SET01-CT01,train_patient_names=SET01-CT02')
        train_test_split_method, train_test_split_paras = parse_train_test_split_str('fixed_ratio,ratio=0.8')
        train_patient_table, test_patient_table = train_test_split(
            table, 
            method = train_test_split_method,
            paras = train_test_split_paras
            )
        train_patient_filenames = get_filenames_from_table(train_patient_table, dataset_path='/u/jw4hv/subspace/onedrive-j-rclone/Dataset/CRT_TOS_Data_Jerry')
    if test_module_name == 'save_prediction':
        import matplotlib.pyplot as plt
        import numpy.ma as ma
        import numpy as np
        # Set Path
        mat_files_path = '/u/jw4hv/subspace/Research_projects/cardiac/cardiac-segmentation/codeV2/exp_results/2022-01-16-scar-seg-TransUNet-maskout-wandb/wandb/run-20220116_222057-tz7tuxrj/files/predictions'
        # List files
        from pathlib import Path
        filefullnames = list(Path(mat_files_path).glob('*.mat'))
        # for filefullname in 
        f0 = filefullnames[0]
        import scipy.io as sio
        mat = sio.loadmat(str(f0), struct_as_record=False, squeeze_me=True)
        img_type = 'img_PSIR'
        mask_pred_type = 'mask_pred'
        mask_gt_type = 'scar_mask'

        img = mat[img_type]
        mask_pred_raw = mat[mask_pred_type]
        mask_pred = np.argmax(mask_pred_raw, axis=-1) == 1
        mask_gt = mat[mask_gt_type]

        # https://stackoverflow.com/questions/57576686/how-to-overlay-segmented-image-on-top-of-main-image-in-python

        xnorm = lambda img: (img - np.min(img)) / (np.max(img) - np.min(img))
        # https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.label2rgb
        # %%
        from skimage import color
        masked_image = color.label2rgb(
            mask_pred, xnorm(img), 
            # alpha = 0.1,
            bg_label = 0,
            image_alpha = 1,
            colors = ['blue'])

        # https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.mark_boundaries
        from skimage import segmentation
        masked_image = segmentation.mark_boundaries(xnorm(img), mask_pred, mode='thick')
        
        
        fig, axs = plt.subplots(1,3)
        # ts0 = train_set[0]
        axs[0].imshow(img, cmap='gray')
        axs[1].imshow(mask_pred, cmap='gray')        
        # axs[2].imshow(ma.array(img, mask = mask_pred), cmap='gray')
        axs[2].imshow(masked_image)

        fig, axs = plt.subplots(1,3)
        # ts0 = train_set[0]
        axs[0].imshow(img, cmap='gray')
        axs[1].imshow(mask_pred, cmap='gray')        
        # axs[2].imshow(ma.array(img, mask = mask_pred), cmap='gray')
        axs[2].imshow(masked_image)

        def Dice(mask1, mask2):
            # return 2*sum(sum(mask1 * mask2)) / (sum(sum(mask1 + mask2)) - sum(sum(mask1*mask2)))
            return 2*(mask1*mask2).sum() / (mask1.sum()+mask2.sum())
        # %%
        import matplotlib as mpl
        masked_pred_img = color.label2rgb(
            mask_pred, xnorm(img), 
            alpha = 0.7,
            bg_label = 0,
            image_alpha = 1,
            colors = ['red'])
        masked_gt_img = color.label2rgb(
            mask_gt, xnorm(img), 
            # alpha = 0.1,
            bg_label = 0,
            image_alpha = 1,
            colors = ['blue'])
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(img, cmap='gray')
        axs[1].imshow(masked_gt_img)
        # axs[2].imshow(ma.array(img, mask = mask_pred), cmap='gray')
        axs[2].imshow(masked_pred_img)
        # axs[2].imshow(img, cmap='gray')
        # axs[2].imshow(np.ma.masked_where(mask_pred<0.5,mask_pred), cmap=mpl.cm.jet_r, alpha=0.5)
        fig.suptitle(f'{mat["Slice Name"]} Dice = {Dice(mask_pred, mask_gt):.5f}', fontsize=16, y=0.8)
        
        # %%
        # training_input_image_grid_masked_GT = \
        #                         torchvision.utils.draw_segmentation_masks(
        #                             image = (training_input_image_grid*255).to(torch.uint8),
        #                             masks = training_mask_GT_grid>0.5,
        #                             alpha = 0.5,
        #                             colors = 'blue')
        import torch, torchvision
        masked_pred_img = torchvision.utils.draw_segmentation_masks(
            # image = (xnorm(img)*255).to(torch.uint8),
            image = torch.from_numpy(np.repeat(xnorm(img)[None,...], 3, axis=0)*255).to(torch.uint8),
            masks = torch.from_numpy(mask_pred>0.5),
            alpha = 0.7,
            colors = 'red'
        )
        masked_gt_img = torchvision.utils.draw_segmentation_masks(
            # image = (xnorm(img)*255).to(torch.uint8),
            image = torch.from_numpy(np.repeat(xnorm(img)[None,...], 3, axis=0)*255).to(torch.uint8),
            masks = torch.from_numpy(mask_gt>0.5),
            alpha = 0.7,
            colors = 'blue'
        )
        
        fig, axs = plt.subplots(1,3)
        axs[0].axis('off')
        axs[0].imshow(img, cmap='gray')        
        axs[0].set_title(img_type)
        # axs[2].imshow(ma.array(img, mask = mask_pred), cmap='gray')
        axs[1].imshow(masked_pred_img.moveaxis(0,-1))
        axs[1].set_title('Prediction')
        axs[1].axis('off')
        axs[2].imshow(masked_gt_img.moveaxis(0,-1))
        axs[2].set_title('Ground Truth')
        axs[2].axis('off')
        # axs[2].imshow(img, cmap='gray')
        # axs[2].imshow(np.ma.masked_where(mask_pred<0.5,mask_pred), cmap=mpl.cm.jet_r, alpha=0.5)
        fig.suptitle(f'{mat["Slice Name"]} Dice = {Dice(mask_pred, mask_gt):.5f}', fontsize=16, y=0.8)
        save_fig_filename = mat['Set Name'] + '-' + mat['Patient Name'] + '-' + mat['Slice Name'] + '.png'
        fig.savefig(str(Path(mat_files_path, save_fig_filename)), bbox_inches='tight')

        # %%
        # Get prediction filenames
        from pathlib import Path
        mat_files_path = r'/u/jw4hv/subspace/Research_projects/cardiac/cardiac-segmentation/codeV2/exp_results/2022-01-16-scar-seg-UNet-maskout-wandb\wandb\run-20220116_181646-dlsbbieq\files\predictions'.replace('\\', '/')
        filefullnames = list(Path(mat_files_path).glob('*.mat'))

        # Load and save files
        import scipy.io as sio
        import torch, torchvision
        img_type = 'img_PSIR'
        mask_pred_type = 'mask_pred'
        mask_gt_type = 'scar_mask'
        for filefullname in filefullnames:            
            # Load data
            mat = sio.loadmat(filefullname, struct_as_record=False, squeeze_me=True)            
            img = mat[img_type]
            mask_pred_raw = mat[mask_pred_type]
            mask_pred = np.argmax(mask_pred_raw, axis=-1) == 1
            mask_gt = mat[mask_gt_type]

            # https://stackoverflow.com/questions/57576686/how-to-overlay-segmented-image-on-top-of-main-image-in-python
            # Generate masked images
            xnorm = lambda img: (img - np.min(img)) / (np.max(img) - np.min(img))
            masked_pred_img = torchvision.utils.draw_segmentation_masks(
                image = torch.from_numpy(np.repeat(xnorm(img)[None,...], 3, axis=0)*255).to(torch.uint8),
                masks = torch.from_numpy(mask_pred>0.5),
                alpha = 0.7,
                colors = 'red'
            )
            masked_gt_img = torchvision.utils.draw_segmentation_masks(
                image = torch.from_numpy(np.repeat(xnorm(img)[None,...], 3, axis=0)*255).to(torch.uint8),
                masks = torch.from_numpy(mask_gt>0.5),
                alpha = 0.7,
                colors = 'blue'
            )

            # Plot and save
            plt.ioff()
            fig, axs = plt.subplots(1,3)
            axs[0].axis('off')
            axs[0].imshow(img, cmap='gray')        
            axs[0].set_title(img_type)
            # axs[2].imshow(ma.array(img, mask = mask_pred), cmap='gray')
            axs[1].imshow(masked_pred_img.moveaxis(0,-1))
            axs[1].set_title('Prediction')
            axs[1].axis('off')
            axs[2].imshow(masked_gt_img.moveaxis(0,-1))
            axs[2].set_title('Ground Truth')
            axs[2].axis('off')
            # axs[2].imshow(img, cmap='gray')
            # axs[2].imshow(np.ma.masked_where(mask_pred<0.5,mask_pred), cmap=mpl.cm.jet_r, alpha=0.5)
            fig.suptitle(f'{mat["Slice Name"]} Dice = {Dice(mask_pred, mask_gt):.5f}', fontsize=16, y=0.8)
            
            save_fig_filename = mat['Set Name'] + '-' + mat['Patient Name'] + '-' + mat['Slice Name'] + '.png'
            fig.savefig(str(Path(mat_files_path, save_fig_filename)), bbox_inches='tight', dpi=250)
