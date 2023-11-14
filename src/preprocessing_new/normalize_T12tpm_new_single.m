%-----------------------------------------------------------------------
% Job saved on 09-Apr-2019 15:51:09 by cfg_util (rev $Rev: 6942 $)
% spm SPM - SPM12 (7219)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------

% addpath /usr/local/spm12/
% ASL->PET->T1, T2-FLAIR->T1, T1->TPM

% load presaved parameters
load('normalize_T12tpm.mat')
matlabbatch = {matlabbatch{1}};

path = '/data/jiahong/data/FDG_PET_preprocessed/';
case_id_list = dir(path);

contrast_list = ["T1_GRE", "T1_SE", "T1c_GRE", "T1c_SE", "T2_FLAIR", "T2_FLAIR_2D", ...
                 "ASL", "DWI", "GRE", "PET_MAC", "PET_QCLEAR", "PET_TOF"]

subj = 'Fdg_Stanford_026';
    
    
disp(strcat('Started ', subj))

% do not process if no T1             
if isfile(strcat(path, subj, '/T1_GRE.nii')) == 1
    ref_path = '/T1_GRE.nii,1';
elseif isfile(strcat(path, subj, '/T1_SE.nii')) == 1
    ref_path = '/T1_SE.nii,1';
else
    disp(strcat('No T1 for', subj))
end

% check if all coregisterations are done
coreg_completed = 1;
normalize_completed = 1;
for contrast_name = contrast_list
    % do not process reference image
    contrast_name_path = char(strcat('/', contrast_name, '.nii,1'));
    if strcmp(contrast_name_path, ref_path)
        continue
    end

    % if contrast does not exist
    if isfile(strcat(path, subj, '/', contrast_name, '.nii')) == 0
        continue
    end

    % check if exists coregistered nii
    if isfile(strcat(path, subj, '/r2T1_', contrast_name, '.nii')) == 0
        coreg_completed = 0;
        break
    end

    % check if exists normalized nii
    if isfile(strcat(path, subj, '/tpm_r2T1_', contrast_name, '.nii')) == 0
        normalize_completed = 0;
        break
    end
end


% normalize all contrasts to template space
resample_list = {};
for contrast_name = contrast_list
    contrast_name_path = char(strcat('/', contrast_name, '.nii,1'));
    contrast_name_r2T1_path = char(strcat('/r2T1_', contrast_name, '.nii,1'));

    % add reference image
    if strcmp(contrast_name_path, ref_path)
        resample_list{end+1} = strcat(path, subj, ref_path);
    end

    % check if this contrast exists
    if isfile(strcat(path, subj, '/r2T1_', contrast_name, '.nii')) == 1
        resample_list{end+1} = strcat(path, subj, contrast_name_r2T1_path);
    end
end

matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.tpm = {'/data/jiahong/data/TPM.nii'};
matlabbatch{1}.spm.spatial.normalise.estwrite.subj.vol = {strcat(path, subj, ref_path)};
matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.prefix = 'tpm_';
matlabbatch{1}.spm.spatial.normalise.estwrite.subj.resample = resample_list';

spm_jobman('run', matlabbatch)

disp(strcat('Finished ', subj))



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
