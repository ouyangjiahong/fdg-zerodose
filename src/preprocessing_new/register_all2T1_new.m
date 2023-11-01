%-----------------------------------------------------------------------
% Job saved on 09-Apr-2019 15:51:09 by cfg_util (rev $Rev: 6942 $)
% spm SPM - SPM12 (7219)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------

% addpath /usr/local/spm12/
% ASL->PET->T1, T2-FLAIR->T1, T1->TPM

% load presaved parameters (have 3 batches, remove the last two, otherwise will have error)
load('register_all2T1.mat')
% matlabbatch{2}=[]
% matlabbatch{3}=[]

path = '/data/jiahong/data/FDG_PET_preprocessed/';
case_id_list = dir(path);

contrast_list = ["T1_GRE", "T1_SE", "T1c_GRE", "T1c_SE", "T2_FLAIR", "T2_FLAIR_2D", ...
                 "ASL", "DWI", "GRE", "PET_MAC", "PET_QCLEAR", "PET_TOF"]

count = 0;
for i = 1 : length(case_id_list) 
    subj_raw = case_id_list(i);
    if strcmp(subj_raw.name, '..') || strcmp(subj_raw.name, '.')
        continue
    end
    subj = subj_raw.name;
    disp(strcat('Started ', subj))
    
    % do not process if no T1             
    if isfile(strcat(path, subj, '/T1_GRE.nii')) == 1
        ref_path = '/T1_GRE.nii,1';
    elseif isfile(strcat(path, subj, '/T1_SE.nii')) == 1
        ref_path = '/T1_SE.nii,1';
    else
        disp(strcat('No T1 for', subj))
        break
    end
    
    % register for each contrast to reference
    for contrast_name = contrast_list
        contrast_name_path = char(strcat('/', contrast_name, '.nii,1'));
        
        % do not process reference image
        if strcmp(contrast_name_path, ref_path)
            continue
        end
        
        % check if this contrast exists
        if isfile(strcat(path, subj, '/', contrast_name, '.nii')) == 1
            disp(contrast_name);
            count = count + 1;
            disp(strcat(path, subj, ref_path))
            disp(strcat(path, subj, contrast_name_path))
            matlabbatch{count} = matlabbatch{1};
            matlabbatch{count}.spm.spatial.coreg.estwrite.ref = {strcat(path, subj, ref_path)};
            matlabbatch{count}.spm.spatial.coreg.estwrite.source = {strcat(path, subj, contrast_name_path)};
            matlabbatch{count}.spm.spatial.coreg.estwrite.roptions.prefix = 'r2T1_';
        end
    end
    spm_jobman('run', matlabbatch)
    disp(strcat('Finished ', subj))
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
