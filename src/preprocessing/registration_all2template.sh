#!/bin/bash

# using FSL
# registration
# transferring T2_FLAIR and PET to T1, then T1 to PET. ASL direct to PET. All end up 256x256x89


# fsl path on longo server
# source /usr/share/fsl/5.0/etc/fslconf/fsl.sh
source /usr/local/fsl/etc/fslconf/fsl.sh

declare -a LIST=('Fdg_Stanford_001')


for i in "${LIST[@]}"
do

DST_DIR="../../data/GBM_selected/"$i
echo $DST_DIR
TPM_DIR="../../data/TPM_resliced_1.nii"
# TPM_DIR="../../data/GBM_selected/Fdg_Stanford_001/PET.nii"

echo Getting co-registering params from T1 to Template space for $i ...
/usr/local/fsl/bin/flirt -in $DST_DIR/T1_inv.nii -ref $TPM_DIR -out $DST_DIR/T1_template -omat $DST_DIR/T1_template.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 6  -interp trilinear

echo Applying transforms for T1 $i ...
/usr/local/fsl/bin/flirt -in $DST_DIR/T1_inv.nii -applyxfm -init $DST_DIR/T1_template.mat -out $DST_DIR/T1_inv_tpm -paddingsize 0.0 -interp trilinear -ref $TPM_DIR

# gunzip $DST_DIR/T1_inv.nii.gz
# gunzip $DST_DIR/T2_FLAIR_inv.nii.gz
# gunzip $DST_DIR/ASL_inv.nii.gz
# rm $DST_DIR/T2_FLAIR_nifti.nii.gz
# mkdir $DST_DIR/transform_mat
# mv $DST_DIR/*.mat $DST_DIR/transform_mat

done
