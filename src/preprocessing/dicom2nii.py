'''
save all dicom files into nifti files in the same folder
'''

import dicom2nifti
import os
from glob import glob
import pdb

data_path = '/data/jiahong/data/FDG_PET_selected_new/'
subj_id_list = []
# for i in range(234, 260):
#     subj_id_list.append('case_'+str(i).zfill(4))

subj_id_all_path = glob(os.path.join(data_path, '*'))
for subj_id_path in subj_id_all_path:
    subj_id = os.path.basename(subj_id_path)
    if 'Fdg' in subj_id:
        subj_id_list.append(subj_id)

    # if 'Anonymized' in subj_id:
    #     continue
    # if 'Fdg' in subj_id:
    #     continue
    # subj_id_list.append(subj_id)

series_names = ['T1', 'T1c', 'T2_FLAIR', 'ASL', 'PET']
# series_names = ['T1', 'T1c']

pdb.set_trace()
for subj_id in subj_id_list:
    subj_path = os.path.join(data_path, subj_id)
    for series_name in series_names:
        series_path = os.path.join(subj_path, series_name)
        if not os.path.exists(series_path):
            continue
        nii_path = os.path.join(subj_path, series_name+'.nii')
        try:
            dicom2nifti.dicom_series_to_nifti(series_path, nii_path)
            print('Saved '+nii_path)
        except:
            print('Cannot handle '+nii_path)
print('Finished All!')
