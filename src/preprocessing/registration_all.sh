#!/bin/bash

# using FSL
# registration
# transferring T2_FLAIR and PET to T1, then T1 to PET. ASL direct to PET. All end up 256x256x89


# fsl path on longo server
# source /usr/share/fsl/5.0/etc/fslconf/fsl.sh
source /usr/local/fsl/etc/fslconf/fsl.sh
FSL_PATH="/usr/local/fsl"

declare -a LIST=('Fdg_Stanford_001' 'Fdg_Stanford_003' 'Fdg_Stanford_004'
            'Fdg_Stanford_005' 'Fdg_Stanford_010' 'Fdg_Stanford_011'
            'Fdg_Stanford_012' 'Fdg_Stanford_014' 'Fdg_Stanford_015'
            'Fdg_Stanford_017' '852_06182015' '869_06252015'
            '1496_05272016' '1498_05312016' '1512_06062016' '1549_06232016'
            '1559_06282016' '1572_07052016' '1604_07142016' '1610_07152016'
            '1619_07192016' '2002_02142017' '2010_02212017' '2120_04122017'
            '2142_04242017' '2275_07062017' '2277_07072017' '2284_07112017'
            '2292_07122017' '2295_07132017' '2310_07242017' '2374_08242017'
            '2377_08252017' '2388_08302017')


for i in "${LIST[@]}"
do

DST_DIR="../../data/GBM_selected/"$i
echo $DST_DIR

echo Getting co-registering params in T1 space for $i ...
$FSL_PATH/bin/flirt -in $DST_DIR/T2_FLAIR.nii -ref $DST_DIR/T1.nii -out $DST_DIR/T2_FLAIR_nifti -omat $DST_DIR/T2_FLAIR_nifti.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 6  -interp trilinear

$FSL_PATH/bin/flirt -in $DST_DIR/PET.nii -ref $DST_DIR/T1.nii -out $DST_DIR/PET_nifti -omat $DST_DIR/PET_nifti.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 6  -interp trilinear


echo Getting inverse transform params for $i ...
$FSL_PATH/bin/convert_xfm -omat $DST_DIR/PET_nifti_inv.mat -inverse $DST_DIR/PET_nifti.mat

$FSL_PATH/bin/convert_xfm -omat $DST_DIR/T2_FLAIR_nifti_inv.mat -concat $DST_DIR/PET_nifti_inv.mat $DST_DIR/T2_FLAIR_nifti.mat


echo Applying transforms for $i ...
$FSL_PATH/bin/flirt -in $DST_DIR/T2_FLAIR.nii -applyxfm -init $DST_DIR/T2_FLAIR_nifti_inv.mat -out $DST_DIR/T2_FLAIR_inv -paddingsize 0.0 -interp trilinear -ref $DST_DIR/PET.nii

$FSL_PATH/bin/flirt -in $DST_DIR/T1.nii -applyxfm -init $DST_DIR/PET_nifti_inv.mat -out $DST_DIR/T1_inv -paddingsize 0.0 -interp trilinear -ref $DST_DIR/PET.nii


echo Getting co-registering params from ASL to PET space for $i ...
$FSL_PATH/bin/flirt -in $DST_DIR/ASL.nii -ref $DST_DIR/PET.nii -out $DST_DIR/ASL_nifti -omat $DST_DIR/ASL_nifti_pet.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 6  -interp trilinear

echo Applying transforms for ASL $i ...
$FSL_PATH/bin/flirt -in $DST_DIR/ASL.nii -applyxfm -init $DST_DIR/ASL_nifti_pet.mat -out $DST_DIR/ASL_inv -paddingsize 0.0 -interp trilinear -ref $DST_DIR/PET.nii

gunzip $DST_DIR/T1_inv.nii.gz
gunzip $DST_DIR/T2_FLAIR_inv.nii.gz
gunzip $DST_DIR/ASL_inv.nii.gz
rm $DST_DIR/T2_FLAIR_nifti.nii.gz
rm $DST_DIR/PET_nifti.nii.gz
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
