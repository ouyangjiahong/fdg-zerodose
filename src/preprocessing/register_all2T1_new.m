%-----------------------------------------------------------------------
% Job saved on 09-Apr-2019 15:51:09 by cfg_util (rev $Rev: 6942 $)
% spm SPM - SPM12 (7219)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------

% addpath /usr/local/spm12/
% ASL->PET->T1, T2-FLAIR->T1, T1->TPM

load('register_all2T1.mat')

%for subj_raw = {'Anonymized_018','Anonymized_019','Anonymized_020','Anonymized_021','Anonymized_022'}
%for subj_raw = {'Anonymized_018'}
%for subj_num = 257 : 260
%for subj_num = [258]
% for subj_raw = {'case_0134', 'case_0142'}
% 'case_0173', 'case_0178',
% for subj_raw = {'2007', '2013', '236', '264', '286'}
for subj_raw = {'286'}
    % subj = strcat('case_0',num2str(subj_num))
    subj = cell2mat(subj_raw)
    disp(subj)
    % path = '/data/jiahong/data/FDG_PET_selected/';
    path = '/data/jiahong/data/zerodose-outside-ad-processed/';
    T1_path = '/T1.nii,1';
    T1c_path = '/T1c.nii,1';
    T2_FLAIR_path = '/T2_FLAIR.nii,1';
    ASL_path = '/ASL.nii,1';
    PET_path = '/PET.nii,1';


    % register T1c to T1
    matlabbatch{4}.spm.spatial.coreg.estwrite.ref = {strcat(path,subj,T1_path)};
    matlabbatch{4}.spm.spatial.coreg.estwrite.source = {strcat(path,subj,T1c_path)};
    matlabbatch{4}.spm.spatial.coreg.estwrite.roptions.prefix = 'r2T1_';

    % register T2-FLAIR to T1
    matlabbatch{1}.spm.spatial.coreg.estwrite.ref = {strcat(path,subj,T1_path)};
    matlabbatch{1}.spm.spatial.coreg.estwrite.source = {strcat(path,subj,T2_FLAIR_path)};
    matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.prefix = 'r2T1_';

    % register ASL to PET
    matlabbatch{2}.spm.spatial.coreg.estwrite.ref = {strcat(path,subj,PET_path)};
    matlabbatch{2}.spm.spatial.coreg.estwrite.source = {strcat(path,subj,ASL_path)};
    matlabbatch{2}.spm.spatial.coreg.estwrite.roptions.prefix = 'r2PET_';

    % register PET to T1
    matlabbatch{3}.spm.spatial.coreg.estwrite.ref = {strcat(path,subj,T1_path)};
    matlabbatch{3}.spm.spatial.coreg.estwrite.source = {strcat(path,subj,PET_path)};
    matlabbatch{3}.spm.spatial.coreg.estwrite.other = {strcat(path,subj,'/r2PET_ASL.nii,1')};
    matlabbatch{3}.spm.spatial.coreg.estwrite.roptions.prefix = 'r2T1_';


    spm_jobman('run',matlabbatch)

    disp(subj)
end


% matlabbatch{1}.spm.spatial.coreg.estwrite.ref = {'/Users/admin/Documents/MATLAB/spm12/tpm/TPM.nii,1'};
% matlabbatch{1}.spm.spatial.coreg.estwrite.source = {'/Users/admin/Documents/jiahong/852_06182015/T1_inv.nii,1'};
% matlabbatch{1}.spm.spatial.coreg.estwrite.other = {
%                                                    '/Users/admin/Documents/jiahong/852_06182015/ASL_inv.nii,1'
%                                                    '/Users/admin/Documents/jiahong/852_06182015/T2_FLAIR_inv.nii,1'
%                                                    '/Users/admin/Documents/jiahong/852_06182015/PET.nii,1'
%                                                    };
% matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
% matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.sep = [4 2];
% matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
% matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.fwhm = [7 7];
% matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.interp = 4;
% matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.wrap = [0 0 0];
% matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.mask = 0;
% matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.prefix = 'r';
