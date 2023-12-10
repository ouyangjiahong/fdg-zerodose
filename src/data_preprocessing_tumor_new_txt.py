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
                    '2374_08242017',
                    'case-0261', 'case-0263', 'case-0266', 'case-0270', 'case-0272', 'case-0284',  # new processed
                    'case-0286', 'case-0287', 'case-0289', 'case-0290', 'case-0295', 'case-0299']
# demantia_subj_list = ['case_0106', 'case_0123', 'case_0130', 'case_0135', 'case_0137', 'case_0141',
#                     'case_0170', 'case_0184', 'case_0193', 'case_0195', 'case_0198', 'case_0209',
#                     'case_0215', 'case_0216', 'case_0224', 'Fdg_Stanford_013']

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
subj_id_list_complete_2 = []
subj_id_list_complete_2_gre = []
subj_id_list_complete_3 = []
subj_id_list_complete_3_gre = []
subj_id_list_complete_4 = []
subj_id_list_complete_4_gre = []
subj_id_list_complete_6 = []
subj_id_list_complete_6_gre = []
subj_all_dict = {}

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
    if not ('PET_MAC' in subj_dict or 'PET_QCLEAR' in subj_dict or 'PET_TOF' in subj_dict):
        continue
    if not ('T1_GRE' in subj_dict or 'T1_SE' in subj_dict):
        continue
    if not ('T2_FLAIR' in subj_dict):
        continue
    subj_id_list_complete_2.append(subj_id)
    if 'T1_GRE' in subj_dict:
        subj_id_list_complete_2_gre.append(subj_id)
    if not ('T1c_GRE' in subj_dict or 'T1c_SE' in subj_dict):
        continue
    subj_id_list_complete_3.append(subj_id)
    if 'T1_GRE' in subj_dict and 'T1c_GRE' in subj_dict:
        subj_id_list_complete_3_gre.append(subj_id)
    if not ('ASL' in subj_dict):
        continue
    subj_id_list_complete_4.append(subj_id)
    if 'T1_GRE' in subj_dict and 'T1c_GRE' in subj_dict:
        subj_id_list_complete_4_gre.append(subj_id)
    if not ('DWI' in subj_dict and 'GRE' in subj_dict):
        continue
    subj_id_list_complete_6.append(subj_id)
    if 'T1_GRE' in subj_dict and 'T1c_GRE' in subj_dict:
        subj_id_list_complete_6_gre.append(subj_id)

print('Total:', len(subj_id_list))
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

print('Total Complete 2:', len(subj_id_list_complete_2))
print('Total Complete 2 GRE:', len(subj_id_list_complete_2_gre))
print('Total Complete 3:', len(subj_id_list_complete_3))
print('Total Complete 3 GRE:', len(subj_id_list_complete_3_gre))
print('Total Complete 4:', len(subj_id_list_complete_4))
print('Total Complete 4 GRE:', len(subj_id_list_complete_4_gre))
print('Total Complete 6:', len(subj_id_list_complete_6))
print('Total Complete 6 GRE:', len(subj_id_list_complete_6_gre))

# tumor_subj_list = subj_id_list
subj_id_list_complete_tumor_2 = np.intersect1d(subj_id_list_complete_2, tumor_subj_list)
subj_id_list_complete_tumor_2_gre = np.intersect1d(subj_id_list_complete_2_gre, tumor_subj_list)
subj_id_list_complete_tumor_3 = np.intersect1d(subj_id_list_complete_3, tumor_subj_list)
subj_id_list_complete_tumor_3_gre = np.intersect1d(subj_id_list_complete_3_gre, tumor_subj_list)
subj_id_list_complete_tumor_4 = np.intersect1d(subj_id_list_complete_4, tumor_subj_list)
subj_id_list_complete_tumor_4_gre = np.intersect1d(subj_id_list_complete_4_gre, tumor_subj_list)
subj_id_list_complete_tumor_6 = np.intersect1d(subj_id_list_complete_6, tumor_subj_list)
subj_id_list_complete_tumor_6_gre = np.intersect1d(subj_id_list_complete_6_gre, tumor_subj_list)
print('Tumor T1/T2-FLAIR:', len(subj_id_list_complete_tumor_2), len(subj_id_list_complete_tumor_2_gre))
print('Tumor T1/T1c/T2-FLAIR:', len(subj_id_list_complete_tumor_3), len(subj_id_list_complete_tumor_3_gre))
print('Tumor T1/T1c/T2-FLAIR/ASL:', len(subj_id_list_complete_tumor_4), len(subj_id_list_complete_tumor_4_gre))
print('Tumor T1/T1c/T2-FLAIR/ASL/DWI/GRE:', len(subj_id_list_complete_tumor_6), len(subj_id_list_complete_tumor_6_gre))
pdb.set_trace()


def save_data_txt(path, subj_id_list, skip=1):
    count = 0
    with open(path, 'w') as ft:
        for subj_id in subj_id_list:
            for i in range(20, 156-20, skip):
                ft.write(subj_id+' '+str(i)+'\n')
                count += 1
    print(count)

def save_data_txt_allslices(path, subj_id_list, skip=1):
    count = 0
    with open(path, 'w') as ft:
        for subj_id in subj_id_list:
            for i in range(0, 156, skip):
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
subj_id_list_sel = subj_id_list_complete_tumor_4_gre
# subj_id_list_sel = subj_id_list_complete_tumor_2_gre
print(len(subj_id_list_sel))
num_subj = len(subj_id_list_sel)

np.random.seed(10)
np.random.shuffle(subj_id_list_sel)


# test_subj_list = ['case_0118', '1619_07192016', 'case_0149', 'Fdg_Stanford_029', 'Fdg_Stanford_016',
#                 'Fdg_Stanford_011', 'case_0168', '1604_07142016', 'Fdg_Stanford_024', 'Fdg_Stanford_027', 'case_0144',
#                 'case_0248', 'case-0284']
test_subj_list = ['case_0149', 'case_0168', 'case_0248', 'case-0284']
pdb.set_trace()
train_subj_list = list(set(subj_id_list_sel) - set(test_subj_list))
test_subj_list = np.intersect1d(subj_id_list_sel, test_subj_list)
print(len(train_subj_list), len(test_subj_list))

# save_data_txt_allslices('../data_new/train_tumor_complete_4_gre_allslices.txt', train_subj_list)
# save_data_txt('../data_new/train_tumor_complete_4_gre.txt', train_subj_list)
# save_data_txt_allslices('../data_new/test_tumor_complete_4_gre_allslices_skip2.txt', test_subj_list, skip=2)
# save_data_txt('../data_new/test_tumor_complete_4_gre_skip2.txt', test_subj_list, skip=2)

save_data_txt_allslices('../data_new/test_tumor_complete_4_both_se_gre_allslices_skip2.txt', test_subj_list, skip=2)
save_data_txt('../data_new/test_tumor_complete_4_both_se_gre_skip2.txt', test_subj_list, skip=2)

# save_data_txt_allslices('../data_new/train_all_complete_2_gre_allslices.txt', train_subj_list)
# save_data_txt('../data_new/train_all_complete_2_gre.txt', train_subj_list)
# save_data_txt_allslices('../data_new/test_all_complete_2_gre_allslices.txt', test_subj_list)
# save_data_txt('../data_new/test_all_complete_2_gre.txt', test_subj_list)

# for fold in range(5):
#     if fold == 4:
#         subj_id_list_test = subj_id_list_sel[fold*int(0.2*num_subj):]
#     else:
#         subj_id_list_test = subj_id_list_sel[fold*int(0.2*num_subj):(fold+1)*int(0.2*num_subj)]
#     # print(subj_id_list_test)
#     # print(len(subj_id_list_test))
#     subj_id_list_train_val = list(set(subj_id_list_sel) - set(subj_id_list_test))
#     # subj_id_list_train_val = subj_id_list_sel[:fold*int(0.2*num_subj)] + subj_id_list_sel[(fold+1)*int(0.2*num_subj):]
#     subj_id_list_val = subj_id_list_train_val[:int(0.1*len(subj_id_list_train_val))]
#     subj_id_list_train = subj_id_list_train_val[int(0.1*len(subj_id_list_train_val)):]
#
#     pdb.set_trace()
#     save_data_txt('../data/fold'+str(fold)+'_train_tumor_complete_nottemplate.txt', subj_id_list_train)
#     save_data_txt('../data/fold'+str(fold)+'_val_tumor_complete_nottemplate.txt', subj_id_list_val)
#     save_data_txt('../data/fold'+str(fold)+'_test_tumor_complete_nottemplate.txt', subj_id_list_test)
#
#     save_data_txt_allslices('../data/fold'+str(fold)+'_test_tumor_complete_allslices_nottemplate.txt', subj_id_list_test)

    # save_data_txt('../data/fold'+str(fold)+'_train_tumor_complete.txt', subj_id_list_train)
    # save_data_txt('../data/fold'+str(fold)+'_val_tumor_complete.txt', subj_id_list_val)
    # save_data_txt('../data/fold'+str(fold)+'_test_tumor_complete.txt', subj_id_list_test)
    #
    # save_data_txt_allslices('../data/fold'+str(fold)+'_test_tumor_complete_allslices.txt', subj_id_list_test)
    # save_data_txt_3d('../data/fold'+str(fold)+'_test_tumor_complete_3d.txt', subj_id_list_test)

# save_data_txt_3d('../data/tumor_complete_3d_all.txt', subj_id_list_sel)
