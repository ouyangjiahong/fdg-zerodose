#!/bin/bash

# Get brainmask from FreeSurfer
#recon-all -subject $1 -i /data3/Amyloid/$1/mr_nifti/T1_nifti_inv.nii -autorecon1 -cw256
cd $SUBJECTS_DIR/$1/mri
mri_label2vol --seg $SUBJECTS_DIR/$1/mri/aparc+aseg.mgz --regheader $SUBJECTS_DIR/$1/mri/aparc+aseg.mgz --o $SUBJECTS_DIR/$1/mri/labels_inv.nii --temp $SUBJECTS_DIR/$1/mri/rawavg.mgz 
cp $SUBJECTS_DIR/$1/mri/labels_inv.nii /data3/Amyloid/$1/mr_nifti/
# Reslice to PET dimensions and convert to Nifti
#mri_vol2vol --mov brainmask.mgz --regheader --targ rawavg.mgz --o brainmask_inv.mgz --no-save-reg
#mri_convert brainmask_inv.mgz brainmask.nii.gz
