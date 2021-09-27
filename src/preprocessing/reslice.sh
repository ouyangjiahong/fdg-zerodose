#!/bin/bash

# using Free Surfer
# reslice PET nii files to 256 x 256 x 89 (1.17, 1.17, 2.78)
# move original PET to PET_raw.nii, resliced file named PET.nii

# first set up environment
# export FREESURFER_HOME=/Applications/freesurfer
# source $FREESURFER_HOME/SetUpFreeSurfer.sh

declare -a LIST=('Fdg_Stanford_001' 'Fdg_Stanford_003' 'Fdg_Stanford_004'
            'Fdg_Stanford_005' 'Fdg_Stanford_010' 'Fdg_Stanford_011'
            'Fdg_Stanford_012' 'Fdg_Stanford_014' 'Fdg_Stanford_015'
            'Fdg_Stanford_017')


for i in "${LIST[@]}"
do
DST_DIR="../../data/GBM_selected/"$i
mv $DST_DIR/PET.nii $DST_DIR/PET_raw.nii

mri_convert -oni 320 -onj 320 -onk 256 $DST_DIR/PET_raw.nii $DST_DIR/PET_tmp1.nii
mri_convert -ois 1.17 -ojs 1.17 -oks 2.78 $DST_DIR/PET_tmp1.nii $DST_DIR/PET_tmp2.nii
mri_convert -oni 256 -onj 256 -onk 89 $DST_DIR/PET_tmp2.nii $DST_DIR/PET.nii

rm $DST_DIR/*tmp*.nii

echo finish reslicing for $i

done
