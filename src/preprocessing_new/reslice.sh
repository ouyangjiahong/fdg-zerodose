#!/bin/bash

# using Free Surfer
# reslice PET nii files to 256 x 256 x 89 (1.17, 1.17, 2.78)
# move original PET to PET_raw.nii, resliced file named PET.nii

# first set up environment
# export FREESURFER_HOME=/usr/local/freesurfer
# source $FREESURFER_HOME/SetUpFreeSurfer.sh


path="/data/jiahong/data/FDG_PET_preprocessed"

for DST_DIR in "$path"/*; do
  if [ -d "$DST_DIR" ]; then
    # Find whether there exists a file called PET_MAC.nii
    if [ -e "$DST_DIR/PET_MAC.nii" ]; then
      # If the file exists, reslice
      mri_convert -oni 320 -onj 320 -onk 256 $DST_DIR/PET_MAC.nii $DST_DIR/PET_tmp1.nii
      mri_convert -ois 1.17 -ojs 1.17 -oks 2.78 $DST_DIR/PET_tmp1.nii $DST_DIR/PET_tmp2.nii
      mri_convert -oni 256 -onj 256 -onk 89 $DST_DIR/PET_tmp2.nii $DST_DIR/reslice_PET_MAC.nii
      rm $DST_DIR/PET_tmp1.nii
      rm $DST_DIR/PET_tmp2.nii
      echo finish reslicing PET_MAC for $DST_DIR
    fi
    if [ -e "$DST_DIR/PET_QCLEAR.nii" ]; then
      # If the file exists, reslice
      mri_convert -oni 320 -onj 320 -onk 256 $DST_DIR/PET_QCLEAR.nii $DST_DIR/PET_tmp1.nii
      mri_convert -ois 1.17 -ojs 1.17 -oks 2.78 $DST_DIR/PET_tmp1.nii $DST_DIR/PET_tmp2.nii
      mri_convert -oni 256 -onj 256 -onk 89 $DST_DIR/PET_tmp2.nii $DST_DIR/reslice_PET_QCLEAR.nii
      rm $DST_DIR/PET_tmp1.nii
      rm $DST_DIR/PET_tmp2.nii
      echo finish reslicing PET_QCLEAR for $DST_DIR
    fi
    if [ -e "$DST_DIR/PET_TOF.nii" ]; then
      # If the file exists, reslice
      mri_convert -oni 320 -onj 320 -onk 256 $DST_DIR/PET_TOF.nii $DST_DIR/PET_tmp1.nii
      mri_convert -ois 1.17 -ojs 1.17 -oks 2.78 $DST_DIR/PET_tmp1.nii $DST_DIR/PET_tmp2.nii
      mri_convert -oni 256 -onj 256 -onk 89 $DST_DIR/PET_tmp2.nii $DST_DIR/reslice_PET_TOF.nii
      rm $DST_DIR/PET_tmp1.nii
      rm $DST_DIR/PET_tmp2.nii
      echo finish reslicing PET_TOF for $DST_DIR
    fi
  fi
done

