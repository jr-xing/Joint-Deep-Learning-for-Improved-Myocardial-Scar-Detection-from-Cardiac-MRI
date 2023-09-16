# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 09:07:05 2021

@author: Jerry
"""
# %% 0.1 Import libraries
#!%reload_ext autoreload
#!%autoreload 2

#%%
import torch
import numpy as np
import scipy.io as sio
from torch._C import Value
from torch.utils.data import Dataset as TorchDataset
from pathlib import Path
import pandas as pd
class Dataset(TorchDataset):
    def __init__(self, data: list, input_info=None, output_info=None, precision=np.float32):
        # , device = torch.device("cpu")
        super(Dataset, self).__init__()
        # if input_types is None or output_types is None:
        #     # self.data = data
        #     self.data = [{data_type: datum[data_type] for data_type in input_types + output_types} for datum in data]
        #     input_types = None
        #     output_types = None
        # else:
        input_types = [info['type'] for info in input_info]
        output_types = [info['type'] for info in output_info]
        self.data = [{data_type: datum[data_type] for data_type in input_types + output_types} for datum in data]
        
        # for other_type in ['augmented', 'patient_name', 'slice_name']:
        #     for datum_idx in range(len(data)):
        #         self.data[datum_idx][other_type] = data[datum_idx][other_type]
            
        self.N = len(data)
        self.input_types = input_types
        self.output_types = output_types

        # Remove the N dimension to make sure the dimension is correct when getting batch data
        # (Depreciated since using mixed precision)(would it work?)
        for key in self.data[0].keys():
            if type(self.data[0][key]) is np.ndarray:
                for datum in self.data:
                    # datum[key] = datum[key][0,:].astype(precision)
                    datum[key] = datum[key].astype(precision)

        # self.device = device
        # if 'strainMat'

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # return self.data[idx]
        # return {key: self.data[idx][key][0, :] for key in self.data[idx].keys()}
        # return self.data[idx]
        return {'img_PSIR': self.data[idx]['img_PSIR'][0],
                'scar_mask': self.data[idx]['scar_mask'][0]}
    
class DatasetSeg(TorchDataset):
    # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
    def __init__(self, 
        data: list, img_data_type=None, mask_data_type=None, precision=np.float32, transform=None, preprocesses=None,
        must_load_myocardium_mask = True, myocardium_mask_type='myocardium_mask'):
        # data should be list of dict
        super().__init__()
        self.preprocesses = preprocesses
        self.transform = transform
        self.img_data_type = img_data_type
        self.mask_data_type = mask_data_type
        self.myocardium_mask_type = myocardium_mask_type
        self.must_load_myocardium_mask = must_load_myocardium_mask
        self.precision = precision
        self.data = data.to_dict('records')
            
        self.N = len(data)


    def load_datum(self, filename):
        mat = sio.loadmat(str(filename), struct_as_record=False, squeeze_me=True)
        image = mat[self.img_data_type]
        mask = mat[self.mask_data_type]        

        datum = {
            self.img_data_type: self.precision(image),
            self.mask_data_type: self.precision(mask)
            }
        if self.must_load_myocardium_mask and self.myocardium_mask_type != self.mask_data_type:
            datum[self.myocardium_mask_type] = mat[self.myocardium_mask_type]
        return datum
    
    def preprocessing(self, datum_ori):
    # def preprocessing(self, datum):
        from utils.data_processing import get_fixed_size_box, crop_img_given_box, resize_image, combine_img_mask
        from utils.data_processing import auto_contrast_adjust
        from utils.data_processing import perform_processing
        # print('before current preprocessing', datum_ori[self.img_data_type].shape)
        datum = perform_processing(datum_ori, self.preprocesses, {'image': self.img_data_type, 'mask': self.mask_data_type})
        # print('after current preprocessing', datum[self.img_data_type].shape)
        return datum

        
    
    def load_all_data(self, dataset_path,
        set_name_key = 'Set Name',
        patient_name_key = 'Patient Name',
        data_path_key = 'LGE Data Path under Patient Directory',
        preprocess = True):

        for datum_idx, datum in enumerate(self.data):
            datum_filename = str(Path(dataset_path,                                   
                                   datum[set_name_key],
                                   datum[patient_name_key],
                                   datum[data_path_key]))
            loaded_datum = self.load_datum(datum_filename)
            self.data[datum_idx].update(loaded_datum)
            if preprocess:
                self.data[datum_idx] = self.preprocessing(self.data[datum_idx])

                
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # print(self.data[idx].keys())
        if self.data is None:
            # print('self.data is None')
            datum = self.load_datum(self.data_filenames[idx])
            datum = self.preprocessing(datum)
            image = datum[self.img_data_type]
            mask = datum[self.mask_data_type]
        else:
            # print('self.data is not None')
            image = self.data[idx][self.img_data_type]
            mask = self.data[idx][self.mask_data_type]
        # print(image.shape, mask.shape)    
        # print('after preprocessing', image.shape, mask.shape)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        # print(image.shape, mask.shape)    
        # print('getitem', image.shape, mask.shape)
        return {'image': image,
                'mask': mask}

class DatasetMTL(TorchDataset):
    def __init__(self, 
        data: list, data_info: list, precision=np.float32, transform=None, preprocesses=None,
        must_load_myocardium_mask = True, myocardium_mask_type='myocardium_mask'):
        # data should be list of dict
        # data_info: 
        #   list of dicts of data, e.g. 
        #   [{'role':'image', 'type': 'img_PSIR'), {'role': 'mask1', 'type': 'myocardium_mask'}]
        # transforms: list of dicts containing target data types and method
        #   e.g. ['targets':['mask1', 'mask2'], 'method': ...]
        super().__init__()
        self.preprocesses = preprocesses
        
        self.data_info = data_info        
        self.output_data_info = [data_info for data_info in self.data_info if data_info.get('feed_to_network', True)]
        self.myocardium_mask_type = myocardium_mask_type
        self.must_load_myocardium_mask = must_load_myocardium_mask
        
        # https://albumentations.ai/docs/examples/example_multi_target/
        self.transform = transform
        if self.transform is None:
            self.transform = lambda x: x
        else:
            transform_additional_targets = {}
            for data_info in self.data_info:
                if data_info['role'] in ['mask', 'mask0', 'mask1', 'mask2']:
                    transform_additional_targets[data_info['role']] = 'mask'
                elif data_info['role'] in ['image']:
                    transform_additional_targets[data_info['role']] = 'image'
            # self.transform.additional_targets = transform_additional_targets
            self.transform.add_targets(transform_additional_targets)

        self.precision = precision
        if type(data) is pd.DataFrame:
            self.data = data.to_dict('records')
        else:
            self.data = data

        # Set used roles for later precision convertion
        self.used_roles = [data_info['role'] for data_info in self.data_info if data_info.get('feed_to_network', True)]
        self.image_dtype = torch.float32
        self.used_image_roles = [role for role in ['image'] if role in self.used_roles]

        self.mask_dtype = torch.long
        self.used_mask_roles = [role for role in ['mask', 'mask1', 'mask2', 'mask3'] if role in self.used_roles]
            
        self.N = len(data)

    def load_datum(self, filename):
        datum = {}
        mat = sio.loadmat(str(filename), struct_as_record=False, squeeze_me=True)

        for data_info in self.data_info:
            if data_info['role'] in ['image', 'mask', 'mask1', 'mask2']:
                datum[data_info['type']] = self.precision(mat[data_info['type']])

        if self.must_load_myocardium_mask and self.myocardium_mask_type not in [data_info['type'] for data_info in self.data_info]:
            datum[self.myocardium_mask_type] = mat[self.myocardium_mask_type]
        return datum

    def preprocessing(self, datum_ori):
        from utils.data_processing import get_fixed_size_box, crop_img_given_box, resize_image, combine_img_mask
        from utils.data_processing import auto_contrast_adjust
        from utils.data_processing import perform_processing
        # print('before current preprocessing', datum_ori[self.img_data_type].shape)
        datum = perform_processing(datum_ori, self.preprocesses)
        # print('after current preprocessing', datum[self.img_data_type].shape)
        return datum

    def preprocesse_all(self):
        for datum_idx in range(len(self.data)):
            self.data[datum_idx] = self.preprocessing(self.data[datum_idx])
    
    def load_all_data(self, dataset_path=None,
        set_name_key = 'Set Name',
        patient_name_key = 'Patient Name',
        data_path_key = 'LGE Data Path under Patient Directory',
        preprocess = True):
        if dataset_path is None:
            from utils.data_io import get_onedrive_path
            dataset_path = str(Path(get_onedrive_path(), 
                          'Dataset/CRT_TOS_Data_Jerry'))

        for datum_idx, datum in enumerate(self.data):
            datum_filename = str(Path(dataset_path,                                   
                                   datum[set_name_key],
                                   datum[patient_name_key],
                                   datum[data_path_key]))
            loaded_datum = self.load_datum(datum_filename)
            self.data[datum_idx].update(loaded_datum)
            if preprocess:
                self.data[datum_idx] = self.preprocessing(self.data[datum_idx])

                
    def __len__(self):
        return self.N
    
    # def __getitemJIT__(self,idx):
    #     # print(self.data[idx].keys())
    #     if self.data is None:
    #         # print('self.data is None')
    #         datum = self.load_datum(self.data_filenames[idx])
    #         datum = self.preprocessing(datum)
    #         image = datum[self.img_data_type]
    #         mask = datum[self.mask_data_type]
    #     else:
    #         print('self.data is not None')
    #     # https://albumentations.ai/docs/examples/example_multi_target/

    #     image = self.data[idx][self.img_data_type]
    #     mask = self.data[idx][self.mask_data_type]
    #     print(image.shape, mask.shape)    
    #     print('after preprocessing', image.shape, mask.shape)
        # if self.transform is not None:
    
    def __getitem__(self, idx):
        output_data = dict(
            [
                (output_data_info['role'], self.data[idx][output_data_info['type']]) for output_data_info in self.output_data_info
                ]
            )
        # print(output_data.keys())
        transformed = self.transform(**output_data)

        # Convert data type
        for image_role in self.used_image_roles:
            transformed[image_role] = transformed[image_role].to(dtype=self.image_dtype)
        for mask_role in self.used_mask_roles:
            transformed[mask_role] = transformed[mask_role].to(dtype=self.mask_dtype)
        # print(image.shape, mask.shape)    
        # print('getitem', image.shape, mask.shape)
        return transformed

#%%
if __name__ == '__main__':
    # test_unit = 'fake_data'
    # test_unit = 'load_data_filenames'
    test_unit = 'preprocessing'
    if test_unit == 'fake_data':
        import numpy as np
        fake_image = np.random.randn(1, 32, 32)
        fake_mask = np.zeros((1, 32, 32))
        fake_mask[:, 10:20, 15:17] = 1
        fake_data = [
            {'img_PSIR': fake_image, 'scar_mask': fake_mask,'myocardium_mask': fake_mask*2}]*5
        fake_data_info = [
            {'role': 'image', 'type': 'img_PSIR'}, 
            {'role': 'mask1', 'type': 'myocardium_mask'},
            {'role': 'mask2', 'type': 'scar_mask'}]

        import albumentations as A
        from albumentations.pytorch import ToTensorV2    
        train_transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=100, p=0.9),
                ToTensorV2(),
            ]
        )
        
        dataset = DatasetMTL(
            data = fake_data, 
            data_info = fake_data_info, 
            transform=train_transform)
        dataset0 = dataset[0]
    elif test_unit == 'load_data_Dataframe':
        from utils.data_io import load_records_from_table, get_onedrive_path
        from pathlib import Path
        dataset_path = str(Path(get_onedrive_path(), 
                          'Dataset/CRT_TOS_Data_Jerry'))        
        # dataset_path = str(Path(
        #     '/home/jrxing/WorkSpace/Research/Cardiac/Dataset/CRT_TOS_Data_Jerry'
        # ))
        table_filename = str(Path(dataset_path, 'record_sheets',
                            'cardiac-LGE-dataset-2022-01-02-two-step-scar-segmentation-fix.xlsx'))        
        
        LGE_data = load_records_from_table(
            data_records_filename=str(table_filename), 
            allow_all=False, 
            must_allowed_for_myo_seg = True,
            must_allowed_for_scar_seg = True,
            must_has_scar=True, 
            must_has_scar_mask=True,
            # not_unsure = True,
            n_read = -1,
            to_dict = False)
        
        from utils.data_io import train_test_split
        from utils.load_config import load_config_from_json
        config = load_config_from_json('./configs/default_crossstitch_myo_scar_seg_config.json')
        train_test_split_config = config['train-test-split']
        training_data, test_data = train_test_split(LGE_data, train_test_split_config['method'], train_test_split_config['paras'])
        
        
        # import albumentations as A
        # from albumentations.pytorch import ToTensorV2    
        # train_transform = A.Compose(
        #     [
        #         A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=100, p=0.9),
        #         ToTensorV2(),
        #     ]
        # )
        
        from utils.dataset import DatasetSeg, DatasetMTL
        # dataset_precision = np.float16
        dataset_precision = np.float32

        # img_data_type = config['data']['image_type']
        # mask_data_type = config['data']['mask_type']
        
        # Pre-processing function
        data_preprocessing_methods = config['preprocessing']
        #%% Augmantation
        # Border types
        # https://docs.opencv.org/4.x/d2/de8/group__core__array.html
        from utils.data_processing import convert_transform_config_to_func
        train_transform = convert_transform_config_to_func(config['train-transform'])
        val_transform = convert_transform_config_to_func(config['validate-transform'])    

        # fake_data = [
        #     {'img_PSIR': fake_image, 'scar_mask': fake_mask,'myocardium_mask': fake_mask*2}]*5
        # fake_data_info = [
        #     {'role': 'image', 'type': 'img_PSIR'}, 
        #     {'role': 'mask1', 'type': 'myocardium_mask'},
        #     {'role': 'mask2', 'type': 'scar_mask'}]
        # dataset = DatasetMTL(
        #     data = fake_data, 
        #     data_info = fake_data_info, 
        #     transform=train_transform)

        # Create Dataset
        train_set = DatasetMTL(
            data = training_data, 
            data_info=config['data']['inputs'], 
            transform=train_transform,
            precision=dataset_precision,
            preprocesses=data_preprocessing_methods)
        val_set = DatasetMTL(
            test_data, 
            data_info=config['data']['inputs'], 
            transform = val_transform, 
            precision=dataset_precision,
            preprocesses=data_preprocessing_methods)
        n_train = len(train_set)
        n_val = len(val_set)
        # train_set0 = train_set[0]
        # if config['training'].get('preload data', False):
        train_set.load_all_data(dataset_path=dataset_path)
        val_set.load_all_data(dataset_path=dataset_path)
        
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
            # axs[2].imshow(ma.array(ts0['image'].numpy()[0], mask = ts0['mask1'].numpy()), cmap='gray')
    elif test_unit == 'preprocessing':
        from utils.load_config import load_config_from_json
        class args: 
            config_file='./configs/test_segmentation_config.json'
        config = load_config_from_json(args.config_file)
        config['data']['mask_type'] = 'scar_mask'
        from utils.data_io import load_records_from_table, get_onedrive_path
        import sys, platform
        from pathlib import Path
        
        # dataset_path = str(Path(get_onedrive_path(), 
        #                       'Documents/Study/Researches/Projects/cardiac/Dataset/CRT_TOS_Data_Jerry'))        
        dataset_path = str(Path(get_onedrive_path(), 
                            'Dataset/CRT_TOS_Data_Jerry'))        
        # dataset_path = str(Path(
        #     '/home/jrxing/WorkSpace/Research/Cardiac/Dataset/CRT_TOS_Data_Jerry'
        # ))
        table_filename = str(Path(dataset_path, 'record_sheets',
                            'cardiac-LGE-dataset-2022-01-02-two-step-scar-segmentation-fix.xlsx'))        
        
        LGE_data = load_records_from_table(
            data_records_filename=str(table_filename), 
            allow_all=False, 
            must_allowed_for_myo_seg = True,
            must_allowed_for_scar_seg = True,
            must_has_scar=True, 
            must_has_scar_mask=True,
            # not_unsure = True,
            n_read = -1,
            to_dict = False)
        from utils.data_io import train_test_split
        # train_test_split_config = {
        #     'method': 'by_patient',
        #     'paras':{
        #         # 'test_patient_names': ['SET01-CT13']
        #         'test_patient_names': ['Pre_CRT_LBBB_with_scar-114_42_BC_MR']
        #         }
        #     }
        train_test_split_config = config['train-test-split']
        training_data, test_data = train_test_split(LGE_data, train_test_split_config['method'], train_test_split_config['paras'])

        from utils.dataset import DatasetSeg
        # dataset_precision = np.float16
        dataset_precision = np.float32

        img_data_type = config['data']['image_type']
        # mask_data_type = 'scar_mask_cropped'
        # mask_data_type = 'myocardium_mask'
        mask_data_type = config['data']['mask_type']
            
        data_preprocessing_methods = config['preprocessing']
        #%% Augmantation
        # Border types
        # https://docs.opencv.org/4.x/d2/de8/group__core__array.html
        from utils.data_processing import convert_transform_config_to_func
        train_transform = convert_transform_config_to_func(config['train-transform'])
        val_transform = convert_transform_config_to_func(config['validate-transform'])

        # %%
        # Create Dataset
        train_set = DatasetSeg(
            training_data, 
            img_data_type=img_data_type, 
            mask_data_type=mask_data_type, 
            transform=train_transform,
            precision=dataset_precision,
            preprocesses=data_preprocessing_methods)
        val_set = DatasetSeg(
            test_data, 
            img_data_type=img_data_type, 
            mask_data_type=mask_data_type,
            transform = val_transform, 
            precision=dataset_precision,
            preprocesses=data_preprocessing_methods)
        n_train = len(train_set)
        n_val = len(val_set)

        # Load all data if using small dataset
        if config['training'].get('preload data', False):
            train_set.load_all_data(dataset_path=dataset_path)
            val_set.load_all_data(dataset_path=dataset_path)
        
        check_loaded = True
        if check_loaded:
            import matplotlib.pyplot as plt
            import numpy.ma as ma
            fig, axs = plt.subplots(1,3)
            ts0 = train_set[0]            
            axs[0].imshow(ts0['image'][0], cmap='gray')
            axs[1].imshow(ts0['mask'], cmap='gray')        
            axs[2].imshow(ma.array(ts0['image'].numpy()[0], mask = ts0['mask'].numpy()), cmap='gray')
            vs0 = val_set[0]
            fig, axs = plt.subplots(1,3)
            axs[0].imshow(vs0['image'][0], cmap='gray')
            axs[1].imshow(vs0['mask'], cmap='gray')        
            axs[2].imshow(ma.array(vs0['image'].numpy()[0], mask = vs0['mask'].numpy()), cmap='gray')