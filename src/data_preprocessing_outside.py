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
# z_score_norm = True
# postfix = 'outside-ad'
postfix = 'outside-ad-complete'
# postfix = 'stanford-ad'
# postfix = 'stanford-ad-complete'

data_path = '/data/jiahong/data/zerodose-outside-ad-processed/'
# data_path = '/data/jiahong/data/FDG_PET_selected_checked/'
subj_paths = glob.glob(data_path+'*')

# sel_subj_list = ['2007', '2013', '236', '264', '286']
# sel_subj_list = ['case_0106', 'case_0123', 'case_0130', 'case_0135', 'case_0137', 'case_0141',
                    # 'case_0170', 'case_0184', 'case_0193', 'case_0195', 'case_0198', 'case_0209',
#                     'case_0215', 'case_0216', 'case_0224', 'Fdg_Stanford_013']
sel_subj_list = ['264', '286']
# sel_subj_list = ['Fdg_Stanford_013']

PET_list = []
T1_list = []
T1c_list = []
T2_FLAIR_list = []
ASL_list = []
subj_id_list = []
subj_id_list_complete = []
subj_all_dict = {}
for subj_path in subj_paths:
    subj_id = os.path.basename(subj_path)
    if len(os.listdir(subj_path)) == 0:
        continue
    subj_id_list.append(subj_id)
    subj_dict = {}
    if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_PET.nii')):
        subj_dict['PET'] = os.path.join(subj_path, 'tpm_r2T1_PET.nii')
        PET_list.append(subj_path)
    if os.path.exists(os.path.join(subj_path, 'tpm_T1.nii')):
        subj_dict['T1'] = os.path.join(subj_path, 'tpm_T1.nii')
        T1_list.append(subj_path)
    if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_T1c.nii')):
        subj_dict['T1c'] = os.path.join(subj_path, 'tpm_r2T1_T1c.nii')
        T1c_list.append(subj_path)
    if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_T2_FLAIR.nii')):
        subj_dict['T2_FLAIR'] = os.path.join(subj_path, 'tpm_r2T1_T2_FLAIR.nii')
        T2_FLAIR_list.append(subj_path)
    if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_r2PET_ASL.nii')):
        subj_dict['ASL'] = os.path.join(subj_path, 'tpm_r2T1_r2PET_ASL.nii')
        ASL_list.append(subj_path)
    subj_all_dict[subj_id] = subj_dict
    # if len(subj_dict) == 5:
    if 'T1' in subj_dict and 'T2_FLAIR' in subj_dict:
        subj_id_list_complete.append(subj_id)

print('Total:', len(subj_id_list))
print('Total Complete:', len(subj_id_list_complete))
print('T1:', len(T1_list))
print('T1c:', len(T1c_list))
print('T2_FLAIR:', len(T2_FLAIR_list))
print('ASL:', len(ASL_list))
print('PET:', len(PET_list))
subj_id_list_complete_sel = np.intersect1d(subj_id_list_complete, sel_subj_list)
print('Selected Complete:', len(subj_id_list_complete_sel))


brain_mask_nib = nib.load(os.path.join('/data/jiahong/fdg-zerodose/data/', 'tpm_mask_new.nii'))
brain_mask = brain_mask_nib.get_fdata()

if z_score_norm == True:
    h5_path = '/data/jiahong/fdg-zerodose/data/'+postfix+'_zscore.h5'
else:
    h5_path = '/data/jiahong/fdg-zerodose/data/'+postfix+'_mean.h5'
h5_path_norm = '/data/jiahong/fdg-zerodose/data/'+postfix+'_norm.h5'

f  = h5py.File(h5_path, 'a')
f_norm  = h5py.File(h5_path_norm, 'a')

subj_data_dict = {}
subj_id_list_save = []
for i, subj_id in enumerate(subj_id_list_complete_sel):
    subj_data = f.create_group(subj_id)
    subj_dict = subj_all_dict[subj_id]
    subj_data_norm = f_norm.create_group(subj_id)
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
        else:
            img  = img / norm

        subj_data.create_dataset(contrast_name, data=img)
        subj_data_norm.create_dataset(contrast_name, data=[norm, std])
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
subj_id_list_save = subj_id_list_complete_sel
subj_id_list_sel = subj_id_list_save
print(len(subj_id_list_complete_sel), len(subj_id_list_sel))
pdb.set_trace()
num_subj = len(subj_id_list_sel)

# save_data_txt_allslices('../data/'+postfix+'_complete_allslices.txt', ['264', '286'])
save_data_txt_allslices('../data/'+postfix+'_allslices.txt', subj_id_list_sel)
