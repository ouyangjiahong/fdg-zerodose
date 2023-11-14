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

subj_list = {"2120_04122017", "8310_01072019", "30453_10012018", "30804_11012018", "Anonymized_018", ...
            "Anonymized_020", "Anonymized_021", "Anonymized_022", "case_0110", "case_0111", "case_0125", ... 
            "case_0129", "case_0131", "case_0174", "case_0184", "case_0198", "case_0205", "case_0219", ...
            "Fdg_Stanford_004", "Fdg_Stanford_006", "Fdg_Stanford_016", "Fdg_Stanford_023", "Fdg_Stanford_024", ...
            "Fdg_Stanford_025", "Fdg_Stanford_026", "Fdg_Stanford_027", "Fdg_Stanford_028", "Fdg_Stanford_029", ...
            "Fdg_Stanford_031", "Fdg_Stanford_032"}
contrast_name = 'ASL';

for subj = subj_list
    subj = char(subj{1});
    disp(strcat('Started ', subj))

    % do not process if no T1             
    if isfile(strcat(path, subj, '/T1_GRE.nii')) == 1
        ref_path = '/T1_GRE.nii,1';
    elseif isfile(strcat(path, subj, '/T1_SE.nii')) == 1
        ref_path = '/T1_SE.nii,1';
    else
        disp(strcat('No T1 for', subj))
        continue
    end

    contrast_name_path = char(strcat('/', contrast_name, '.nii,1'));
    matlabbatch{1}.spm.spatial.coreg.estwrite.ref = {strcat(path, subj, ref_path)};
    matlabbatch{1}.spm.spatial.coreg.estwrite.source = {strcat(path, subj, contrast_name_path)};
    matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.prefix = 'r2T1_';

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
