%-----------------------------------------------------------------------
% Job saved on 09-Apr-2019 15:51:09 by cfg_util (rev $Rev: 6942 $)
% spm SPM - SPM12 (7219)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------

% addpath /usr/local/spm12/
% ASL->PET->T1, T2-FLAIR->T1, T1->TPM

load('register_all2T1.mat')

%  'Fdg_Stanford_001', 'Fdg_Stanford_003', 'Fdg_Stanford_004',...
%             'Fdg_Stanford_005', 'Fdg_Stanford_010', 'Fdg_Stanford_011',...
%             'Fdg_Stanford_014', 'Fdg_Stanford_015',...
%             'Fdg_Stanford_017', ...
%             '1610_07152016',...
%             '2002_02142017'
%for subj = {'Fdg_Stanford_012','852_06182015', '869_06252015',...
%            '1496_05272016', '1498_05312016', '1512_06062016', '1549_06232016',...
%            '1559_06282016', '1572_07052016', '1604_07142016',...
%            '1619_07192016', '2010_02212017', '2120_04122017',...
%            '2142_04242017', '2275_07062017', '2277_07072017', '2284_07112017',...
%            '2292_07122017', '2295_07132017', '2310_07242017', '2374_08242017',...
%            '2377_08252017', '2388_08302017'}
%for subj_raw = {'Anonymized_018','Anonymized_019','Anonymized_020','Anonymized_021','Anonymized_022'}
%for subj_raw = {'Anonymized_018'}
%for subj_num = 257 : 260
for subj_num = [240, 241, 242, 248, 251, 252, 257, 258]
    subj = strcat('case_0',num2str(subj_num))
    % subj = cell2mat(subj_raw)
    disp(subj)
    path = '/data/jiahong/data/FDG_PET_selected_new/';
    T1_path = '/T1.nii,1';
    T1c_path = '/T1c.nii,1';
    T2_FLAIR_path = '/T2_FLAIR.nii,1';
    ASL_path = '/ASL.nii,1';
    PET_path = '/PET.nii,1';

    % do not process if do not have T1 or PET
    % if exist(strcat(path,subj,'/T1.nii'))~=2 || exist(strcat(path,subj,'/PET.nii'))~=2
    if exist(strcat(path,subj,'/T1.nii'))~=2
        continue
    end

    % register T2-FLAIR to T1
    if exist(strcat(path,subj,'/T2_FLAIR.nii')) == 2
    matlabbatch{1}.spm.spatial.coreg.estwrite.ref = {strcat(path,subj,T1_path)};
    matlabbatch{1}.spm.spatial.coreg.estwrite.source = {strcat(path,subj,T2_FLAIR_path)};
    matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.prefix = 'r2T1_';
    end

    % register T1c to T1
    if exist(strcat(path,subj,'/T1c.nii')) == 2
    matlabbatch{4}.spm.spatial.coreg.estwrite.ref = {strcat(path,subj,T1_path)};
    matlabbatch{4}.spm.spatial.coreg.estwrite.source = {strcat(path,subj,T1c_path)};
    matlabbatch{4}.spm.spatial.coreg.estwrite.roptions.prefix = 'r2T1_';
    end

    % register ASL to PET
    if exist(strcat(path,subj,'/ASL.nii')) == 2
    matlabbatch{2}.spm.spatial.coreg.estwrite.ref = {strcat(path,subj,PET_path)};
    matlabbatch{2}.spm.spatial.coreg.estwrite.source = {strcat(path,subj,ASL_path)};
    matlabbatch{2}.spm.spatial.coreg.estwrite.roptions.prefix = 'r2PET_';
    end

    % register PET (ASL) to T1
    if exist(strcat(path,subj,'/ASL.nii')) == 2
    matlabbatch{3}.spm.spatial.coreg.estwrite.ref = {strcat(path,subj,T1_path)};
    matlabbatch{3}.spm.spatial.coreg.estwrite.source = {strcat(path,subj,PET_path)};
    matlabbatch{3}.spm.spatial.coreg.estwrite.other = {strcat(path,subj,'/r2PET_ASL.nii,1')};
    matlabbatch{3}.spm.spatial.coreg.estwrite.roptions.prefix = 'r2T1_';
    else
    matlabbatch{3}.spm.spatial.coreg.estwrite.ref = {strcat(path,subj,T1_path)};
    matlabbatch{3}.spm.spatial.coreg.estwrite.source = {strcat(path,subj,PET_path)};
    matlabbatch{3}.spm.spatial.coreg.estwrite.roptions.prefix = 'r2T1_';
    end

    spm_jobman('run',matlabbatch)

%     register T1 (ASL, PET, T2_FLAIR) to template
%     matlabbatch{1}.spm.spatial.coreg.estwrite.ref = {template_path};
%     matlabbatch{1}.spm.spatial.coreg.estwrite.source = {strcat(path,subj,T1_path)};
%     matlabbatch{1}.spm.spatial.coreg.estwrite.other = {strcat(path,subj,'/rT1_T2_FLAIR.nii,1')
%                                                                strcat(path,subj,'/rT1_rPET_ASL.nii,1')
%                                                                strcat(path,subj,'/rT1_PET.nii,1')};
%     matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.prefix = 'rTPM_';
%     spm_jobman('run',matlabbatch)

%     matlabbatch{1}.spm.spatial.coreg.estwrite.ref = {template_path};
%     matlabbatch{1}.spm.spatial.coreg.estwrite.source = {strcat(path,subj,ASL_path)};
%     matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.prefix = 'rTPM_';
%     spm_jobman('run',matlabbatch)

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
