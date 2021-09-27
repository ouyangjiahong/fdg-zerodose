%-----------------------------------------------------------------------
% Job saved on 11-Apr-2019 15:38:04 by cfg_util (rev $Rev: 6942 $)
% spm SPM - SPM12 (7219)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------

% after register T2_FLAIR, ASL, PET to T1, normalize all to template space
load('normalize_T12tpm.mat')

matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.interp = 4;
matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.reg = [0.2 0.2 0.2 0.2 0.2];
matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.tpm = {'/data/jiahong/data/TPM.nii'};
matlabbatch{1}.spm.spatial.normalise.estwrite.subj.vol = {'/data/jiahong/project_zerodose_pytorch/src/preprocessing/T1.nii,1'};
matlabbatch{1}.spm.spatial.normalise.estwrite.subj.resample = {'/data/jiahong/project_zerodose_pytorch/src/preprocessing/T1.nii,1'};
matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.prefix = 'tpm_new_';
spm_jobman('run',matlabbatch)


% matlabbatch{1}.spm.spatial.normalise.estwrite.subj.vol = {'/Users/admin/Documents/jiahong/GBM_selected/Fdg_Stanford_001/T1.nii,1'};
% matlabbatch{1}.spm.spatial.normalise.estwrite.subj.resample = {
%                                                                '/Users/admin/Documents/jiahong/GBM_selected/Fdg_Stanford_001/r2T1_T2_FLAIR.nii,1'
%                                                                '/Users/admin/Documents/jiahong/GBM_selected/Fdg_Stanford_001/r2T1_PET.nii,1'
%                                                                '/Users/admin/Documents/jiahong/GBM_selected/Fdg_Stanford_001/r2T1_r2PET_ASL.nii,1'
%                                                                '/Users/admin/Documents/jiahong/GBM_selected/Fdg_Stanford_001/T1.nii,1'
%                                                                };
% matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.biasreg = 0.0001;
% matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.biasfwhm = 60;
% matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.tpm = {'/Users/admin/Documents/MATLAB/spm12/tpm/TPM.nii'};
% matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.affreg = 'mni';
% matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.reg = [0 0.001 0.5 0.05 0.2];
% matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.fwhm = 0;
% matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.samp = 3;
% matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.bb = [-78 -112 -70
%                                                              78 76 85];
% matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.vox = [1 1 1];
% matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.interp = 4;
% matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.prefix = 'tpm_';
