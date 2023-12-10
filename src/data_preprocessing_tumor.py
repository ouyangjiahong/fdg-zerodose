import os
import glob
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
import scipy.ndimage
from datetime import datetime
import matplotlib.pyplot as plt
import pdb

z_score_norm = False
max_score_norm = True

# data_path = '/data/jiahong/data/FDG_PET_selected_checked/'
data_path = '/data/jiahong/data/FDG_PET_selected_all/'
subj_paths = glob.glob(data_path+'*')

# tumor_subj_list = ['case_0103', 'case_0110', 'case_0111', 'case_0117', 'case_0118', 'case_0124',
#                     'case_0129', 'case_0132', 'case_0136', 'case_0139', 'case_0142', 'case_0144',
#                     'case_0145', 'case_0146', 'case_0148', 'case_0149', 'case_0151', 'case_0156',
#                     'case_0157', 'case_0158', 'case_0160', 'case_0161', 'case_0164', 'case_0168',
#                     'case_0169', 'case_0174', 'case_0175', 'case_0188', 'case_0196', 'case_0208',
#                     'case_0219', 'case_0226', 'case_0238', 'case_0240', 'case_0241', 'case_0242',
#                     'case_0248', 'case_0251', 'case_0252', 'case_0257', 'case_0258', 'Fdg_Stanford_001',
#                     'Fdg_Stanford_002', 'Fdg_Stanford_003', 'Fdg_Stanford_004', 'Fdg_Stanford_005',
#                     'Fdg_Stanford_006', 'Fdg_Stanford_010', 'Fdg_Stanford_011', 'Fdg_Stanford_012',
#                     'Fdg_Stanford_014', 'Fdg_Stanford_015', 'Fdg_Stanford_016', 'Fdg_Stanford_017',
#                     'Fdg_Stanford_023', 'Fdg_Stanford_024', 'Fdg_Stanford_025', 'Fdg_Stanford_026',
#                     'Fdg_Stanford_027', 'Fdg_Stanford_028', 'Fdg_Stanford_029', 'Fdg_Stanford_030',
#                     'Fdg_Stanford_031', 'Fdg_Stanford_032', '852_06182015', '869_06252015', '1496_05272016',
#                     '1498_05312016', '1512_06062016', '1549_06232016', '1559_06282016', '1572_07052016',
#                     '1604_07142016', '1610_07152016', '1619_07192016', '1657_08022016', '2002_02142017',
#                     '2010_02212017', '2114_04102017', '2120_04122017', '2142_04242017', '2275_07062017',
#                     '2277_07072017', '2284_07112017', '2292_07122017', '2295_07132017', '2310_07242017',
#                     '2374_08242017', '2377_08252017', '2388_08302017']
# remove multiple scans from the same subjects
tumor_subj_list = ['case_0103', 'case_0110', 'case_0111', 'case_0117', 'case_0118', 'case_0124',
                    'case_0129', 'case_0132', 'case_0136', 'case_0139', 'case_0142', 'case_0144',
                    'case_0145', 'case_0146', 'case_0148', 'case_0149', 'case_0151', 'case_0156',
                    'case_0157', 'case_0158', 'case_0160', 'case_0161', 'case_0164', 'case_0168',
                    'case_0169', 'case_0174', 'case_0175', 'case_0188', 'case_0196', 'case_0208',
                    'case_0219', 'case_0226', 'case_0238', 'case_0240', 'case_0241', 'case_0242',
                    'case_0248', 'case_0251', 'case_0252', 'case_0257', 'case_0258', 'Fdg_Stanford_001',
                    'Fdg_Stanford_002', 'Fdg_Stanford_003', 'Fdg_Stanford_004', 'Fdg_Stanford_005',
                    'Fdg_Stanford_006', 'Fdg_Stanford_010', 'Fdg_Stanford_011', 'Fdg_Stanford_012',
                    'Fdg_Stanford_014', 'Fdg_Stanford_015', 'Fdg_Stanford_016', 'Fdg_Stanford_017',
                    'Fdg_Stanford_023', 'Fdg_Stanford_024', 'Fdg_Stanford_025', 'Fdg_Stanford_026',
                    'Fdg_Stanford_027', 'Fdg_Stanford_028', 'Fdg_Stanford_029', 'Fdg_Stanford_030',
                    'Fdg_Stanford_031', 'Fdg_Stanford_032', '852_06182015', '1496_05272016',
                    '1549_06232016',
                    '1604_07142016', '1619_07192016', '2002_02142017',
                    '2010_02212017', '2120_04122017', '2275_07062017',
                    '2284_07112017',
                    '2374_08242017']
demantia_subj_list = ['case_0106', 'case_0123', 'case_0130', 'case_0135', 'case_0137', 'case_0141',
                    'case_0170', 'case_0184', 'case_0193', 'case_0195', 'case_0198', 'case_0209',
                    'case_0215', 'case_0216', 'case_0224', 'Fdg_Stanford_013']

PET_list = []
T1_list = []
T1c_list = []
T2_FLAIR_list = []
ASL_list = []
subj_id_list = []
subj_id_list_complete = []
subj_all_dict = {}
for subj_id in tumor_subj_list:
    subj_path = os.path.join(data_path, subj_id)
    if len(os.listdir(subj_path)) == 0:
        continue
    subj_id_list.append(subj_id)
    subj_dict = {}
    if os.path.exists(os.path.join(subj_path, 'r2T1_PET.nii')):
        subj_dict['PET'] = os.path.join(subj_path, 'r2T1_PET.nii')
        PET_list.append(subj_path)
    if os.path.exists(os.path.join(subj_path, 'T1.nii')):
        subj_dict['T1'] = os.path.join(subj_path, 'T1.nii')
        T1_list.append(subj_path)
    if os.path.exists(os.path.join(subj_path, 'r2T1_T1c.nii')):
        subj_dict['T1c'] = os.path.join(subj_path, 'r2T1_T1c.nii')
        T1c_list.append(subj_path)
    if os.path.exists(os.path.join(subj_path, 'r2T1_T2_FLAIR.nii')):
        subj_dict['T2_FLAIR'] = os.path.join(subj_path, 'r2T1_T2_FLAIR.nii')
        T2_FLAIR_list.append(subj_path)
    if os.path.exists(os.path.join(subj_path, 'r2T1_r2PET_ASL.nii')):
        subj_dict['ASL'] = os.path.join(subj_path, 'r2T1_r2PET_ASL.nii')
        ASL_list.append(subj_path)
    subj_all_dict[subj_id] = subj_dict
    if len(subj_dict) == 5:
        subj_id_list_complete.append(subj_id)
# for subj_path in subj_paths:
    # subj_id = os.path.basename(subj_path)
    # if len(os.listdir(subj_path)) == 0:
    #     continue
    # subj_id_list.append(subj_id)
    # subj_dict = {}
    # if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_PET.nii')):
    #     subj_dict['PET'] = os.path.join(subj_path, 'tpm_r2T1_PET.nii')
    #     PET_list.append(subj_path)
    # if os.path.exists(os.path.join(subj_path, 'tpm_T1.nii')):
    #     subj_dict['T1'] = os.path.join(subj_path, 'tpm_T1.nii')
    #     T1_list.append(subj_path)
    # if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_T1c.nii')):
    #     subj_dict['T1c'] = os.path.join(subj_path, 'tpm_r2T1_T1c.nii')
    #     T1c_list.append(subj_path)
    # if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_T2_FLAIR.nii')):
    #     subj_dict['T2_FLAIR'] = os.path.join(subj_path, 'tpm_r2T1_T2_FLAIR.nii')
    #     T2_FLAIR_list.append(subj_path)
    # if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_r2PET_ASL.nii')):
    #     subj_dict['ASL'] = os.path.join(subj_path, 'tpm_r2T1_r2PET_ASL.nii')
    #     ASL_list.append(subj_path)
    # subj_all_dict[subj_id] = subj_dict
    # if len(subj_dict) == 5:
    #     subj_id_list_complete.append(subj_id)

print('Total:', len(subj_id_list))
print('Total Complete:', len(subj_id_list_complete))
print('T1:', len(T1_list))
print('T1c:', len(T1c_list))
print('T2_FLAIR:', len(T2_FLAIR_list))
print('ASL:', len(ASL_list))
print('PET:', len(PET_list))
subj_id_list_complete_tumor = np.intersect1d(subj_id_list_complete, tumor_subj_list)
print('Tumor Complete:', len(subj_id_list_complete_tumor))


brain_mask_nib = nib.load(os.path.join('/data/jiahong/fdg-zerodose/data/', 'tpm_mask_new.nii'))
brain_mask = brain_mask_nib.get_fdata()

# if z_score_norm == True:
#     h5_path = '/data/jiahong/fdg-zerodose/data/tumor_complete_zscore.h5'
# elif max_score_norm == True:
#     h5_path = '/data/jiahong/fdg-zerodose/data/tumor_complete_max.h5'
# else:
#     h5_path = '/data/jiahong/fdg-zerodose/data/tumor_complete_mean.h5'
if z_score_norm == True:
    h5_path = '/data/jiahong/fdg-zerodose/data/tumor_complete_zscore_nottemplate.h5'
elif max_score_norm == True:
    h5_path = '/data/jiahong/fdg-zerodose/data/tumor_complete_max_nottemplate.h5'
else:
    h5_path = '/data/jiahong/fdg-zerodose/data/tumor_complete_mean_nottemplate.h5'
# h5_path_norm = '/data/jiahong/fdg-zerodose/data/tumor_complete_norm.h5'

f  = h5py.File(h5_path, 'a')
# f_norm  = h5py.File(h5_path_norm, 'a')

subj_data_dict = {}
subj_id_list_save = []
for i, subj_id in enumerate(subj_id_list_complete_tumor):
    subj_data = f.create_group(subj_id)
    subj_dict = subj_all_dict[subj_id]
    # subj_data_norm = f_norm.create_group(subj_id)
    for contrast_name in subj_dict.keys():
        img_nib = nib.load(subj_dict[contrast_name])
        img = img_nib.get_fdata()
        if img.shape != (157, 189, 156) or np.nanmax(img) == 0 or np.isnan(img[:,:,20:-20]).sum()>100000:
            print(subj_id)
            print(img.shape, np.nanmax(img), np.isnan(img[:,:,20:-20]).sum())
            break
        img = np.nan_to_num(img, nan=0.)
        img = img * brain_mask
        img[img<0] = 0
        img = np.concatenate([img, np.zeros((3,189,156))], 0)     # (157,189) -> (160,192)
        img = np.concatenate([img, np.zeros((160,3,156))], 1)

        norm = img.mean()
        # img = img / norm  # norm by dividing mean
        std = img.std()
        if z_score_norm == True:
            img = (img - norm) / (std + 1e-8)
        elif max_score_norm == True:
            img_max = np.percentile(img, 98)
            img_min = np.percentile(img, 2)
            img = (img - img_min) / (img_max - img_min)
            img = np.clip(img, a_max=1., a_min=0.)
        else:
            img  = img / norm

        subj_data.create_dataset(contrast_name, data=img)
        # subj_data_norm.create_dataset(contrast_name, data=[norm, std])
    subj_id_list_save.append(subj_id)
    print(i, subj_id)


def save_data_txt(path, subj_id_list):
    count = 0
    with open(path, 'w') as ft:
        for subj_id in subj_id_list:
            for i in range(20, 156-20):
                ft.write(subj_id+' '+str(i)+'\n')
                count += 1
    print(count)

def save_data_txt_allslices(path, subj_id_list):
    count = 0
    with open(path, 'w') as ft:
        for subj_id in subj_id_list:
            for i in range(0, 156):
                ft.write(subj_id+' '+str(i)+'\n')
                count += 1
    print(count)

def save_data_txt_3d(path, subj_id_list):
    count = 0
    with open(path, 'w') as ft:
        for subj_id in subj_id_list:
            ft.write(subj_id+'\n')
            count += 1
    print(count)

#
subj_id_list_save = subj_id_list_complete_tumor
subj_id_list_sel = subj_id_list_save
print(len(subj_id_list_complete_tumor), len(subj_id_list_sel))
pdb.set_trace()
num_subj = len(subj_id_list_sel)

np.random.seed(10)
np.random.shuffle(subj_id_list_sel)

for fold in range(5):
    if fold == 4:
        subj_id_list_test = subj_id_list_sel[fold*int(0.2*num_subj):]
    else:
        subj_id_list_test = subj_id_list_sel[fold*int(0.2*num_subj):(fold+1)*int(0.2*num_subj)]
    # print(subj_id_list_test)
    # print(len(subj_id_list_test))
    subj_id_list_train_val = list(set(subj_id_list_sel) - set(subj_id_list_test))
    # subj_id_list_train_val = subj_id_list_sel[:fold*int(0.2*num_subj)] + subj_id_list_sel[(fold+1)*int(0.2*num_subj):]
    subj_id_list_val = subj_id_list_train_val[:int(0.1*len(subj_id_list_train_val))]
    subj_id_list_train = subj_id_list_train_val[int(0.1*len(subj_id_list_train_val)):]

    pdb.set_trace()
    save_data_txt('../data/fold'+str(fold)+'_train_tumor_complete_nottemplate.txt', subj_id_list_train)
    save_data_txt('../data/fold'+str(fold)+'_val_tumor_complete_nottemplate.txt', subj_id_list_val)
    save_data_txt('../data/fold'+str(fold)+'_test_tumor_complete_nottemplate.txt', subj_id_list_test)

    save_data_txt_allslices('../data/fold'+str(fold)+'_test_tumor_complete_allslices_nottemplate.txt', subj_id_list_test)

    # save_data_txt('../data/fold'+str(fold)+'_train_tumor_complete.txt', subj_id_list_train)
    # save_data_txt('../data/fold'+str(fold)+'_val_tumor_complete.txt', subj_id_list_val)
    # save_data_txt('../data/fold'+str(fold)+'_test_tumor_complete.txt', subj_id_list_test)
    #
    # save_data_txt_allslices('../data/fold'+str(fold)+'_test_tumor_complete_allslices.txt', subj_id_list_test)
    # save_data_txt_3d('../data/fold'+str(fold)+'_test_tumor_complete_3d.txt', subj_id_list_test)

# save_data_txt_3d('../data/tumor_complete_3d_all.txt', subj_id_list_sel)
