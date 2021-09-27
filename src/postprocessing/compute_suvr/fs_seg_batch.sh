#!/bin/bash

# Get brainmask from FreeSurfer
#recon-all -subject 011_S_4120_B -i /data3/Amyloid/ADNI_PROJECTS/ADNI_NO_DOSE/011_S_4120_B/T1_nifti_inv_KC.nii -all -cw256
#recon-all -subject 2511 -i /data3/Amyloid/2511/mr_nifti/T1_nifti_inv.nii -all -cw256
#recon-all -subject 2416 -i /data3/Amyloid/2416/mr_nifti/T1_nifti_inv.nii -all -cw256
#recon-all -subject 2425 -i /data3/Amyloid/2425/mr_nifti/T1_nifti_inv.nii -all -cw256
#recon-all -subject 50767 -i /data3/Amyloid/50767/mr_nifti/T1_nifti_inv.nii -all -cw256

recon-all -subject 011_S_4120_B -autorecon2 -autorecon3 -cw256
recon-all -subject 011_S_4222_B -autorecon2 -autorecon3 -cw256
recon-all -subject 011_S_4278_B -autorecon2 -autorecon3 -cw256
recon-all -subject 013_S_4579_B -autorecon2 -autorecon3 -cw256
recon-all -subject 014_S_4401_B -autorecon2 -autorecon3 -cw256
recon-all -subject 014_S_4576_B -autorecon2 -autorecon3 -cw256
recon-all -subject 023_S_4501_B -autorecon2 -autorecon3 -cw256
recon-all -subject 023_S_5120 -autorecon2 -autorecon3 -cw256
recon-all -subject 023_S_5241 -autorecon2 -autorecon3 -cw256
recon-all -subject 024_S_4084_B -autorecon2 -autorecon3 -cw256
#cd $SUBJECTS_DIR/$1/mri

# Reslice to PET dimensions and convert to Nifti (check interp method flags!!)
#mri_vol2vol --mov brainmask.mgz --regheader --targ rawavg.mgz --o brainmask_inv.mgz --no-save-reg
#mri_convert brainmask_inv.mgz brainmask_inv.nii.gz
#mri_label2vol --seg $SUBJECTS_DIR/$1/mri/aparc+aseg.mgz --regheader $SUBJECTS_DIR/$1/mri/aparc+aseg.mgz --o $SUBJECTS_DIR/$1/mri/labels_inv.nii --temp $SUBJECTS_DIR/$1/mri/rawavg.mgz 