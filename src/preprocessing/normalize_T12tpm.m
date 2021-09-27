%-----------------------------------------------------------------------
% Job saved on 11-Apr-2019 15:38:04 by cfg_util (rev $Rev: 6942 $)
% spm SPM - SPM12 (7219)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------

% after register T2_FLAIR, ASL, PET to T1, normalize all to template space
load('normalize_T12tpm.mat')
%   'Fdg_Stanford_011','Fdg_Stanford_014','Fdg_Stanford_003', 'Fdg_Stanford_005', 'Fdg_Stanford_010',...
%              'Fdg_Stanford_015',...
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
% for subj_num = 234 : 260
    % subj = strcat('case_0',num2str(subj_num))
% for subj_raw = {'Anonymized_018','Anonymized_019','Anonymized_020','Anonymized_021','Anonymized_022'}
for subj_raw = {'case_0122', 'case_0124', 'case_0150', 'case_0180', 'case_0235', 'case_0246', 'case_0249', 'case_0250', 'case_0253'}
    subj = cell2mat(subj_raw);
    path = '/data/jiahong/data/FDG_PET_selected/';

    % do not process if do not have T1 or PET
    % if exist(strcat(path,subj,'/T1.nii'))~=2 || exist(strcat(path,subj,'/PET.nii'))~=2
    if exist(strcat(path,subj,'/T1.nii'))~=2
        continue
    end

    % matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.tpm = {'/Users/admin/Documents/MATLAB/spm12/tpm/TPM.nii'};
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.tpm = {'/data/jiahong/data/TPM.nii'};
    matlabbatch{1}.spm.spatial.normalise.estwrite.subj.vol = {strcat(path,subj,'/T1.nii,1')};
    if exist(strcat(path,subj,'/r2T1_T2_FLAIR.nii')) == 2 && exist(strcat(path,subj,'/r2T1_ASL.nii')) == 2 && exist(strcat(path,subj,'/r2T1_T1c.nii')) == 2
        matlabbatch{1}.spm.spatial.normalise.estwrite.subj.resample = {
                                                                    strcat(path,subj,'/r2T1_T1c.nii,1')
                                                                   strcat(path,subj,'/r2T1_T2_FLAIR.nii,1')
                                                                   strcat(path,subj,'/r2T1_PET.nii,1')
                                                                   strcat(path,subj,'/r2T1_r2PET_ASL.nii,1')
                                                                   strcat(path,subj,'/T1.nii,1')
                                                                   };
    elseif exist(strcat(path,subj,'/r2T1_T2_FLAIR.nii')) == 2 && exist(strcat(path,subj,'/r2T1_ASL.nii')) == 2 && exist(strcat(path,subj,'/r2T1_T1c.nii')) ~= 2
       matlabbatch{1}.spm.spatial.normalise.estwrite.subj.resample = {
                                                                  strcat(path,subj,'/r2T1_T2_FLAIR.nii,1')
                                                                  strcat(path,subj,'/r2T1_PET.nii,1')
                                                                  strcat(path,subj,'/r2T1_r2PET_ASL.nii,1')
                                                                  strcat(path,subj,'/T1.nii,1')
                                                                  };
    elseif exist(strcat(path,subj,'/r2T1_T2_FLAIR.nii')) == 2 && exist(strcat(path,subj,'/r2T1_ASL.nii')) ~= 2 && exist(strcat(path,subj,'/r2T1_T1c.nii')) == 2
      matlabbatch{1}.spm.spatial.normalise.estwrite.subj.resample = {
                                                                  strcat(path,subj,'/r2T1_T1c.nii,1')
                                                                 strcat(path,subj,'/r2T1_T2_FLAIR.nii,1')
                                                                 strcat(path,subj,'/r2T1_PET.nii,1')
                                                                 strcat(path,subj,'/T1.nii,1')
                                                                 };
    elseif exist(strcat(path,subj,'/r2T1_T2_FLAIR.nii')) ~= 2 && exist(strcat(path,subj,'/r2T1_ASL.nii')) == 2 && exist(strcat(path,subj,'/r2T1_T1c.nii')) == 2
        matlabbatch{1}.spm.spatial.normalise.estwrite.subj.resample = {
                                                                    strcat(path,subj,'/r2T1_T1c.nii,1')
                                                                   strcat(path,subj,'/r2T1_PET.nii,1')
                                                                   strcat(path,subj,'/r2T1_r2PET_ASL.nii,1')
                                                                   strcat(path,subj,'/T1.nii,1')
                                                                   };
    elseif exist(strcat(path,subj,'/r2T1_T2_FLAIR.nii')) == 2 && exist(strcat(path,subj,'/r2T1_ASL.nii')) ~= 2 && exist(strcat(path,subj,'/r2T1_T1c.nii')) ~= 2
       matlabbatch{1}.spm.spatial.normalise.estwrite.subj.resample = {
                                                                  strcat(path,subj,'/r2T1_T2_FLAIR.nii,1')
                                                                  strcat(path,subj,'/r2T1_PET.nii,1')
                                                                  strcat(path,subj,'/T1.nii,1')
                                                                  };
    elseif exist(strcat(path,subj,'/r2T1_T2_FLAIR.nii')) ~= 2 && exist(strcat(path,subj,'/r2T1_ASL.nii')) == 2 && exist(strcat(path,subj,'/r2T1_T1c.nii')) ~= 2
      matlabbatch{1}.spm.spatial.normalise.estwrite.subj.resample = {
                                                                 strcat(path,subj,'/r2T1_PET.nii,1')
                                                                 strcat(path,subj,'/r2T1_r2PET_ASL.nii,1')
                                                                 strcat(path,subj,'/T1.nii,1')
                                                                 };
    elseif exist(strcat(path,subj,'/r2T1_T2_FLAIR.nii')) ~= 2 && exist(strcat(path,subj,'/r2T1_ASL.nii')) ~= 2 && exist(strcat(path,subj,'/r2T1_T1c.nii')) == 2
     matlabbatch{1}.spm.spatial.normalise.estwrite.subj.resample = {
                                                                 strcat(path,subj,'/r2T1_T1c.nii,1')
                                                                strcat(path,subj,'/r2T1_PET.nii,1')
                                                                strcat(path,subj,'/T1.nii,1')
                                                                };
    else
    matlabbatch{1}.spm.spatial.normalise.estwrite.subj.resample = {
                                                               strcat(path,subj,'/r2T1_PET.nii,1')
                                                               strcat(path,subj,'/T1.nii,1')
                                                               };
    end
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
