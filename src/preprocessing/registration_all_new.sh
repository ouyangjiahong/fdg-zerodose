#!/bin/bash

# using FSL
# registration
# transferring T2_FLAIR and PET to T1, then T1 to PET. ASL direct to PET. All end up 256x256x89


# fsl path on longo server
# source /usr/share/fsl/5.0/etc/fslconf/fsl.sh
source /usr/local/fsl/etc/fslconf/fsl.sh
FSL_PATH="/usr/local/fsl"

declare -a LIST=('2007' '2013' '236' '264' '286')


for i in "${LIST[@]}"
do

DST_DIR="/data/jiahong/data/Jiahong_Zaharchuk_Data/"$i
echo $DST_DIR

echo Getting co-registering params in T1 space for $i ...
$FSL_PATH/bin/flirt -in $DST_DIR/T2_FLAIR.nii -ref $DST_DIR/T1.nii -out $DST_DIR/T2_FLAIR_nifti -omat $DST_DIR/T2_FLAIR_nifti.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 6  -interp trilinear

$FSL_PATH/bin/flirt -in $DST_DIR/PET.nii -ref $DST_DIR/T1.nii -out $DST_DIR/PET_nifti -omat $DST_DIR/PET_nifti.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 6  -interp trilinear

echo Applying transforms for T2-FLAIR and PET $i ...
$FSL_PATH/bin/flirt -in $DST_DIR/T2_FLAIR.nii -applyxfm -init $DST_DIR/T2_FLAIR_nifti.mat -out $DST_DIR/T2_FLAIR_inv -paddingsize 0.0 -interp trilinear -ref $DST_DIR/T1.nii

$FSL_PATH/bin/flirt -in $DST_DIR/PET.nii -applyxfm -init $DST_DIR/PET_nifti.mat -out $DST_DIR/PET_inv -paddingsize 0.0 -interp trilinear -ref $DST_DIR/T1.nii


gunzip $DST_DIR/T2_FLAIR_inv.nii.gz
gunzip $DST_DIR/PET_inv.nii.gz
mkdir $DST_DIR/transform_mat
mv $DST_DIR/*.mat $DST_DIR/transform_mat

# for those T1 and T2_FLAIR very similar to PET

# echo Getting co-registering params from T1 to PET space for $i ...
# /usr/local/fsl/bin/flirt -in $DST_DIR/T1.nii -ref $DST_DIR/PET.nii -out $DST_DIR/T1_nifti -omat $DST_DIR/T1_nifti_pet.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 6  -interp trilinear
#
# echo Applying transforms for T1 $i ...
# /usr/local/fsl/bin/flirt -in $DST_DIR/T1.nii -applyxfm -init $DST_DIR/T1_nifti_pet.mat -out $DST_DIR/T1_inv -paddingsize 0.0 -interp trilinear -ref $DST_DIR/PET.nii
#
# echo Getting co-registering params from T2_FLAIR to PET space for $i ...
# /usr/local/fsl/bin/flirt -in $DST_DIR/T2_FLAIR.nii -ref $DST_DIR/PET.nii -out $DST_DIR/T2_FLAIR_nifti -omat $DST_DIR/T2_FLAIR_nifti_pet.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 6  -interp trilinear
#
# echo Applying transforms for T2_FLAIR $i ...
# /usr/local/fsl/bin/flirt -in $DST_DIR/T2_FLAIR.nii -applyxfm -init $DST_DIR/T2_FLAIR_nifti_pet.mat -out $DST_DIR/T2_FLAIR_inv -paddingsize 0.0 -interp trilinear -ref $DST_DIR/PET.nii
#
# gunzip $DST_DIR/T1_inv.nii.gz
# gunzip $DST_DIR/T2_FLAIR_inv.nii.gz
# rm $DST_DIR/T2_FLAIR_nifti.nii.gz
# rm $DST_DIR/T1_nifti.nii.gz
# rm $DST_DIR/transform_mat/*
# mv $DST_DIR/*.mat $DST_DIR/transform_mat

done
