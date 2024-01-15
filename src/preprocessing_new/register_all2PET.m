%-----------------------------------------------------------------------
% Job saved on 09-Apr-2019 15:51:09 by cfg_util (rev $Rev: 6942 $)
% spm SPM - SPM12 (7219)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------

% addpath /usr/local/spm12/
% ASL->PET->T1, T2-FLAIR->T1, T1->TPM

% load presaved parameters (have 3 batches, remove the last two, otherwise will have error)
load('register_all2T1.mat')
matlabbatch = {matlabbatch{1}};

path = '/data/jiahong/data/FDG_PET_preprocessed/';
case_id_list = dir(path);

contrast_list = ["r2T1_T1c_GRE", "r2T1_T1c_SE", "r2T1_T2_FLAIR", "r2T1_T2_FLAIR_2D", ...
                 "r2T1_ASL", "r2T1_DWI", "r2T1_GRE", "reslice_PET_MAC", "reslice_PET_QCLEAR", "reslice_PET_TOF"]

for i = 1 : length(case_id_list) 
    subj_raw = case_id_list(i);
    if strcmp(subj_raw.name, '..') || strcmp(subj_raw.name, '.')
        continue
    end
    subj = subj_raw.name;
    disp(strcat('Started ', subj))
    
    % do not process if no T1
    if isfile(strcat(path, subj, '/T1_GRE.nii')) == 1
        T1_path = '/T1_GRE.nii,1';
        contrast_T1_list = ["T1_GRE", "r2T1_T1_SE"]
    elseif isfile(strcat(path, subj, '/T1_SE.nii')) == 1
        T1_path = '/T1_SE.nii,1';
        contrast_T1_list = ["T1_SE"]
    else
        disp(strcat('No T1 for ', subj))
        continue
    end
    
    % do not process if no PET            
    if isfile(strcat(path, subj, '/PET_MAC.nii')) == 1
        ref_path = '/reslice_PET_MAC.nii,1';
    elseif isfile(strcat(path, subj, '/PET_QCLEAR.nii')) == 1
        ref_path = '/reslice_PET_QCLEAR.nii,1';
    elseif isfile(strcat(path, subj, '/PET_TOF.nii')) == 1
        ref_path = '/reslice_PET_TOF.nii,1';
    else 
        disp(strcat('No PET for ', subj))
        continue
    end
    
    %{
    % check if all coregisterations are done
    coreg_completed = 1;
    
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
    end
     
    % coregisterations are completed
    if coreg_completed == 1
        disp('coregisterations are completed')
        continue
    end
    %}
    
    %{
    % register for each contrast to reference
    count = 0;
    matlabbatch = {matlabbatch{1}};
    for contrast_name = [contrast_list, contrast_T1_list]
        contrast_name_path = char(strcat('/', contrast_name, '.nii,1'));
        
        % do not process reference image
        if strcmp(contrast_name_path, ref_path)
            continue
        end
        
        % check if this contrast exists
        if isfile(strcat(path, subj, '/', contrast_name, '.nii')) == 1
            disp(contrast_name);
            disp(strcat(path, subj, ref_path))
            disp(strcat(path, subj, contrast_name_path))
            count = count + 1;
            matlabbatch{count} = matlabbatch{1};
            matlabbatch{count}.spm.spatial.coreg.estwrite.ref = {strcat(path, subj, ref_path)};
            matlabbatch{count}.spm.spatial.coreg.estwrite.source = {strcat(path, subj, contrast_name_path)};
            matlabbatch{count}.spm.spatial.coreg.estwrite.roptions.prefix = 'r2PET_';
        end
    end
    %}
    
    resample_list = {};
    for contrast_name = [contrast_list, contrast_T1_list]
        contrast_name_path = char(strcat('/', contrast_name, '.nii,1'));
        
        % do not process reference image, PET
        if strcmp(contrast_name_path, ref_path)
            continue
        end
        
        % check if this contrast exists
        if isfile(strcat(path, subj, '/', contrast_name, '.nii')) == 1
            disp(contrast_name);
            disp(strcat(path, subj, ref_path))
            disp(strcat(path, subj, contrast_name_path))
            resample_list{end+1} = strcat(path, subj, contrast_name_path);
        end
    end
    
    matlabbatch{1}.spm.spatial.coreg.estwrite.ref = {strcat(path, subj, ref_path)};
    matlabbatch{1}.spm.spatial.coreg.estwrite.source = {strcat(path, subj, T1_path)};
    matlabbatch{1}.spm.spatial.coreg.estwrite.other = resample_list';
    matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.prefix = 'r2PET_';

    
    spm_jobman('run', matlabbatch)
    disp(strcat('Finished ', subj))
end

