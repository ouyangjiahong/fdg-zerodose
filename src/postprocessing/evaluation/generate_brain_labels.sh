#!/bin/bash

export FREESURFER_HOME=/usr/local/freesurfer
export SUBJECTS_DIR=output
source /usr/local/freesurfer/SetUpFreeSurfer.sh

$1="case_0101"

recon-all -subject $1 -i data/$1/tpm_T1.nii -autorecon1 -cw256
recon-all -subject $1 -autorecon2 -autorecon3 -cw256
mri_label2vol --seg $SUBJECTS_DIR/$1/mri/aparc+aseg.mgz --regheader $SUBJECTS_DIR/$1/mri/aparc+aseg.mgz --o $SUBJECTS_DIR/$1/mri/labels_inv.nii --temp $SUBJECTS_DIR/$1/mri/rawavg.mgz
