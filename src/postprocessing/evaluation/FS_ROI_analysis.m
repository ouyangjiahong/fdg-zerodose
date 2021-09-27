function ROI_All = FS_ROI_analysis

[Amy_path, params] = Amyloid_pathfinder;

% Get correct label file
cd(fullfile(Amy_path, char(params.subj_ind{1}),'mr_nifti'))
FSlabel_file = 'labels_inv.nii';
labels_hdr = spm_vol(FSlabel_file);
labels = spm_read_vols(labels_hdr);

rois_noMC = unique(labels);
cd(fullfile(Amy_path, char(params.subj_ind{1}),'Full_Dose'))

ROI_All      = zeros(size(rois_noMC,1)-1,size(params.subj_ind,1),4);
%ROI_Hundredth = zeros(size(rois_noMC,1)-1,size(params.subj_ind,1));
%ROI_mc        = zeros(size(rois_noMC,1)-1,size(params.subj_ind,1));
%ROI_petonly   = zeros(size(rois_noMC,1)-1,size(params.subj_ind,1));

for i = 1:size(params.subj_ind,1)
    disp(char(params.subj_ind{i}))
    % Load volumes
    Amy_Full = dicom2nii (fullfile(Amy_path, char(params.subj_ind{i}), 'Full_Dose'),'_bin*.sdcopen','_sl','.sdcopen',fullfile(Amy_path, char(params.subj_ind{1}),'mr_nifti/T1_nifti_inv.nii'), 0);
    Amy_Hundredth = dicom2nii (fullfile(Amy_path, char(params.subj_ind{i}), 'Hundredth_Dose'),'_bin*.sdcopen','_sl','.sdcopen',fullfile(Amy_path, char(params.subj_ind{1}),'mr_nifti/T1_nifti_inv.nii'), 0);
    if i<10
        Amy_mc = dicom2nii (fullfile(Amy_path, 'reading', ['Patient0' num2str(i)], 'mc'),'anon*.dcm','anon_','.dcm',fullfile(Amy_path, char(params.subj_ind{1}),'mr_nifti/T1_nifti_inv.nii'), 0);
        Amy_petonly = dicom2nii (fullfile(Amy_path, 'reading', ['Patient0' num2str(i)], 'petonly'),'anon*.dcm','anon_','.dcm',fullfile(Amy_path, char(params.subj_ind{1}),'mr_nifti/T1_nifti_inv.nii'), 0);
    else
        Amy_mc = dicom2nii (fullfile(Amy_path, 'reading', ['Patient' num2str(i)], 'mc'),'anon*.dcm','anon_','.dcm',fullfile(Amy_path, char(params.subj_ind{1}),'mr_nifti/T1_nifti_inv.nii'), 0);
        Amy_petonly = dicom2nii (fullfile(Amy_path, 'reading', ['Patient' num2str(i)], 'petonly'),'anon*.dcm','anon_','.dcm',fullfile(Amy_path, char(params.subj_ind{1}),'mr_nifti/T1_nifti_inv.nii'), 0);
    end
    % Get correct label file
    cd(fullfile(Amy_path, char(params.subj_ind{i}),'mr_nifti'))
    FSlabel_file = 'labels_inv.nii';
    labels_hdr = spm_vol(FSlabel_file);
    labels = spm_read_vols(labels_hdr);
    
    % Get cerebellum region
    label_cerebel = zeros(size(labels));
    label_cerebel(labels==8 | labels==47) = 1;
    
    % Get SUVRs
    SUVR_Full = Amy_Full/mean(Amy_Full(label_cerebel==1));
    SUVR_Hundredth = Amy_Hundredth/mean(Amy_Hundredth(label_cerebel==1));
    SUVR_mc = Amy_mc/mean(Amy_mc(label_cerebel==1));
    SUVR_petonly = Amy_petonly/mean(Amy_petonly(label_cerebel==1));
    
    % Get regional SUVRs
    for j = 1:size(rois_noMC,1)-1
        ROI_All(j,i,1) = mean(SUVR_Full(labels==rois_noMC(j+1)));
        ROI_All(j,i,2) = mean(SUVR_Hundredth(labels==rois_noMC(j+1)));
        ROI_All(j,i,3) = mean(SUVR_mc(labels==rois_noMC(j+1)));
        ROI_All(j,i,4) = mean(SUVR_petonly(labels==rois_noMC(j+1)));
    end
end