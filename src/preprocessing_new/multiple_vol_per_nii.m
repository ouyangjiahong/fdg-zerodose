% some nifti has two volumes but do not match with the size description, thus cannot open by nibabel
% addpath /usr/local/spm12/

path = '/data/jiahong/data/FDG_PET_preprocessed/Fdg_Stanford_003/tpm_T1_GRE.nii';
path_new = '/data/jiahong/data/FDG_PET_preprocessed/Fdg_Stanford_003/tpm_T1_GRE_new.nii';
path2 = '/data/jiahong/data/FDG_PET_preprocessed/Fdg_Stanford_003/tpm_r2T1_T1_SE.nii';

headerinfo = spm_vol(path);
headerinfo2 = spm_vol(path2);

data = spm_read_vols(headerinfo(1));
headerinfo2.fname = path_new;
headerinfo2.private.dat.fname = headerinfo2.fname;

spm_write_vol(headerinfo2, data);

