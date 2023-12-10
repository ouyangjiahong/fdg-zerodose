%-----------------------------------------------------------------------
% Job saved on 11-Apr-2019 15:38:04 by cfg_util (rev $Rev: 6942 $)
% spm SPM - SPM12 (7219)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------

% after register T2_FLAIR, ASL, PET to T1, normalize all to template space
load('normalize_T12tpm.mat')

% for subj_num = [240, 241, 242, 248, 251, 252, 257, 258]
% for subj_num = 234 : 260
% for subj = subj_list
% for subj_raw = {'852_06182015', '869_06252015',...
  %         '1496_05272016', '1549_06232016',...
%             '1559_06282016', '1572_07052016', '1604_07142016',...
%             '1619_07192016', '2120_04122017',...
%             '2275_07062017', '2277_07072017', '2284_07112017',...
%             '2292_07122017', '2295_07132017', '2310_07242017', '2374_08242017',...
%             '2388_08302017', '2377_08252017', '1610_07152016','2002_02142017'}
% for subj_raw = {'Fdg_Stanford_001', 'Fdg_Stanford_002','Fdg_Stanford_003', 'Fdg_Stanford_004',...
%             'Fdg_Stanford_005', 'Fdg_Stanford_006', 'Fdg_Stanford_007', 'Fdg_Stanford_008', 'Fdg_Stanford_009','Fdg_Stanford_010', 'Fdg_Stanford_011',...
%             'Fdg_Stanford_012', 'Fdg_Stanford_013','Fdg_Stanford_014', 'Fdg_Stanford_015',...
%             'Fdg_Stanford_016', 'Fdg_Stanford_017','Fdg_Stanford_023', 'Fdg_Stanford_024',  ...
%           'Fdg_Stanford_025', 'Fdg_Stanford_026','Fdg_Stanford_027', 'Fdg_Stanford_028',  ...
%         'Fdg_Stanford_029', 'Fdg_Stanford_030','Fdg_Stanford_031', 'Fdg_Stanford_032'}
    % '1498_05312016', '1512_06062016','2114_04102017'
% for subj_raw = {'1498_05312016', '1512_06062016'}
%for subj_raw = {'case_0122', 'case_0124', 'case_0150', 'case_0180', 'case_0235', 'case_0246', 'case_0249', 'case_0250', 'case_0253'}
for subj_raw = {'264', '286'}
    % subj = strcat('case_0',num2str(subj_num))
    subj = cell2mat(subj_raw)
    % subj = char(subj);
    % path = '/data/jiahong/data/FDG_PET_selected_new/';
    path = '/data/jiahong/data/zerodose-outside-ad-processed/';

    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.tpm = {'/data/jiahong/data/TPM.nii'};
    matlabbatch{1}.spm.spatial.normalise.estwrite.subj.vol = {strcat(path,subj,'/T1.nii,1')};
    matlabbatch{1}.spm.spatial.normalise.estwrite.subj.resample = {
                                                              strcat(path,subj,'/r2T1_PET.nii,1')
                                                              strcat(path,subj,'/r2T1_T2_FLAIR.nii,1')
                                                              strcat(path,subj,'/r2T1_r2PET_ASL.nii,1')
                                                              strcat(path,subj,'/T1.nii,1')


                                                                   };




    matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.prefix = 'tpm_';

    spm_jobman('run',matlabbatch)
    disp(subj)
end


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
