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
# data_path = '/data/jiahong/data/FDG_PET_selected_all/'
data_path = '/data/jiahong/data/FDG_PET_preprocessed/'
subj_paths = glob.glob(data_path+'*')


PET_MAC_list = []
PET_QCLEAR_list = []
PET_TOF_list = []
T1_GRE_list = []
T1_SE_list = []
T1c_GRE_list = []
T1c_SE_list = []
T2_FLAIR_list = []
T2_FLAIR_2D_list = []
ASL_list = []
DWI_list = []
GRE_list = []

subj_id_list = []
subj_id_list_complete = []
subj_all_dict = {}
# for subj_id in tumor_subj_list:
#     subj_path = os.path.join(data_path, subj_id)
#     if len(os.listdir(subj_path)) == 0:
#         continue
    # subj_id_list.append(subj_id)
    # subj_dict = {}
    # if os.path.exists(os.path.join(subj_path, 'r2T1_PET.nii')):
    #     subj_dict['PET_MAC'] = os.path.join(subj_path, 'r2T1_PET.nii')
    #     PET_list.append(subj_path)
    # if os.path.exists(os.path.join(subj_path, 'T1.nii')):
    #     subj_dict['T1'] = os.path.join(subj_path, 'T1.nii')
    #     T1_list.append(subj_path)
    # if os.path.exists(os.path.join(subj_path, 'r2T1_T1c.nii')):
    #     subj_dict['T1c'] = os.path.join(subj_path, 'r2T1_T1c.nii')
    #     T1c_list.append(subj_path)
    # if os.path.exists(os.path.join(subj_path, 'r2T1_T2_FLAIR.nii')):
    #     subj_dict['T2_FLAIR'] = os.path.join(subj_path, 'r2T1_T2_FLAIR.nii')
    #     T2_FLAIR_list.append(subj_path)
    # if os.path.exists(os.path.join(subj_path, 'r2T1_r2PET_ASL.nii')):
    #     subj_dict['ASL'] = os.path.join(subj_path, 'r2T1_r2PET_ASL.nii')
    #     ASL_list.append(subj_path)
    # subj_all_dict[subj_id] = subj_dict
    # if len(subj_dict) == 5:
    #     subj_id_list_complete.append(subj_id)
for subj_path in subj_paths:
    subj_id = os.path.basename(subj_path)
    if len(os.listdir(subj_path)) == 0:
        continue
    subj_id_list.append(subj_id)
    subj_dict = {}
    if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_PET_MAC.nii')):
        subj_dict['PET_MAC'] = os.path.join(subj_path, 'tpm_r2T1_PET_MAC.nii')
        PET_MAC_list.append(subj_path)

    if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_PET_QCLEAR.nii')):
        subj_dict['PET_QCLEAR'] = os.path.join(subj_path, 'tpm_r2T1_PET_QCLEAR.nii')
        PET_QCLEAR_list.append(subj_path)

    if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_PET_TOF.nii')):
        subj_dict['PET_TOF'] = os.path.join(subj_path, 'tpm_r2T1_PET_TOF.nii')
        PET_TOF_list.append(subj_path)

    if os.path.exists(os.path.join(subj_path, 'tpm_T1_GRE.nii')):
        subj_dict['T1_GRE'] = os.path.join(subj_path, 'tpm_T1_GRE.nii')
        T1_GRE_list.append(subj_path)

    if os.path.exists(os.path.join(subj_path, 'tpm_T1_SE.nii')):
        subj_dict['T1_SE'] = os.path.join(subj_path, 'tpm_T1_SE.nii')
        T1_SE_list.append(subj_path)
    elif os.path.exists(os.path.join(subj_path, 'tpm_r2T1_T1_SE.nii')):
        subj_dict['T1_SE'] = os.path.join(subj_path, 'tpm_r2T1_T1_SE.nii')
        T1_SE_list.append(subj_path)

    if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_T1c_GRE.nii')):
        subj_dict['T1c_GRE'] = os.path.join(subj_path, 'tpm_r2T1_T1c_GRE.nii')
        T1c_GRE_list.append(subj_path)

    if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_T1c_SE.nii')):
        subj_dict['T1c_SE'] = os.path.join(subj_path, 'tpm_r2T1_T1c_SE.nii')
        T1c_SE_list.append(subj_path)

    if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_T2_FLAIR.nii')):
        subj_dict['T2_FLAIR'] = os.path.join(subj_path, 'tpm_r2T1_T2_FLAIR.nii')
        T2_FLAIR_list.append(subj_path)

    if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_T2_FLAIR_2D.nii')):
        subj_dict['T2_FLAIR_2D'] = os.path.join(subj_path, 'tpm_r2T1_T2_FLAIR_2D.nii')
        T2_FLAIR_2D_list.append(subj_path)

    if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_ASL.nii')):
        subj_dict['ASL'] = os.path.join(subj_path, 'tpm_r2T1_ASL.nii')
        ASL_list.append(subj_path)

    if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_DWI.nii')):
        subj_dict['DWI'] = os.path.join(subj_path, 'tpm_r2T1_DWI.nii')
        DWI_list.append(subj_path)

    if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_GRE.nii')):
        subj_dict['GRE'] = os.path.join(subj_path, 'tpm_r2T1_GRE.nii')
        GRE_list.append(subj_path)

    subj_all_dict[subj_id] = subj_dict
    # if 'PET_MAC' in subj_data or 'PET_QCLEAR':
    #     subj_id_list_complete.append(subj_id)

print('Total:', len(subj_id_list))
print('Total Complete:', len(subj_id_list_complete))
print('T1_GRE:', len(T1_GRE_list))
print('T1_SE:', len(T1_SE_list))
print('T1c_GRE:', len(T1c_GRE_list))
print('T1c_SE:', len(T1c_SE_list))
print('T2_FLAIR:', len(T2_FLAIR_list))
print('T2_FLAIR_2D:', len(T2_FLAIR_2D_list))
print('ASL:', len(ASL_list))
print('DWI:', len(DWI_list))
print('GRE:', len(GRE_list))
print('PET_MAC:', len(PET_MAC_list))
print('PET_QCLEAR:', len(PET_QCLEAR_list))
print('PET_TOF:', len(PET_TOF_list))

pdb.set_trace()

brain_mask_nib = nib.load(os.path.join('/data/jiahong/fdg-zerodose/data/', 'tpm_mask_new.nii'))
brain_mask = brain_mask_nib.get_fdata()

if z_score_norm == True:
    h5_path = '/data/jiahong/fdg-zerodose/data_new/all_cases_complete_zscore.h5'
    h5_path_norm = '/data/jiahong/fdg-zerodose/data_new/all_cases_complete_mean_std_norm.h5'
elif max_score_norm == True:
    h5_path = '/data/jiahong/fdg-zerodose/data_new/all_cases_complete_max.h5'
    h5_path_norm = '/data/jiahong/fdg-zerodose/data_new/all_cases_complete_max_norm.h5'
else:
    h5_path = '/data/jiahong/fdg-zerodose/data_new/all_cases_complete_mean.h5'
    h5_path_norm = '/data/jiahong/fdg-zerodose/data_new/all_cases_complete_mean_std_norm.h5'

# if z_score_norm == True:
#     h5_path = '/data/jiahong/fdg-zerodose/data/tumor_complete_zscore_nottemplate.h5'
# elif max_score_norm == True:
#     h5_path = '/data/jiahong/fdg-zerodose/data/tumor_complete_max_nottemplate.h5'
# else:
#     h5_path = '/data/jiahong/fdg-zerodose/data/tumor_complete_mean_nottemplate.h5'


f  = h5py.File(h5_path, 'a')
f_norm  = h5py.File(h5_path_norm, 'a')

subj_data_dict = {}
subj_id_list_save = []
subj_id_list = sorted(subj_id_list)
for i, subj_id in enumerate(subj_id_list):
    # if subj_id not in ['Fdg_Stanford_002', 'Fdg_Stanford_016', 'case_0137', 'case_0219']:
    #     continue
    # del f[subj_id]
    # del f_norm[subj_id]
    subj_data = f.create_group(subj_id)
    subj_dict = subj_all_dict[subj_id]
    subj_data_norm = f_norm.create_group(subj_id)
    for contrast_name in subj_dict.keys():
        img_nib = nib.load(subj_dict[contrast_name])
        img = img_nib.get_fdata()
        if img.shape != (157, 189, 156) or np.nanmax(img) == 0 or np.isnan(img[:,:,20:-20]).sum()>100000:
            print(subj_id)
            print(img.shape, np.nanmax(img), np.isnan(img[:,:,20:-20]).sum())
            # break
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
        if max_score_norm == True:
            subj_data_norm.create_dataset(contrast_name, data=[img_min, img_max])
        else:
            subj_data_norm.create_dataset(contrast_name, data=[norm, std])

    subj_id_list_save.append(subj_id)
    print(i, subj_id)
