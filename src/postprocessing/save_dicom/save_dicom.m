clear
test_set = {'case_0103', 'case_0110', 'case_0111', 'case_0117', 'case_0118', 'case_0124', ...
                    'case_0129', 'case_0132', 'case_0136', 'case_0139', 'case_0142', 'case_0144',...
                    'case_0145', 'case_0146', 'case_0148', 'case_0149', 'case_0151', 'case_0156',...
                    'case_0157', 'case_0158', 'case_0160', 'case_0161', 'case_0164', 'case_0168',...
                    'case_0169', 'case_0174', 'case_0175', 'case_0188', 'case_0196', 'case_0208',...
                    'case_0219', 'case_0226', 'case_0238', 'case_0240', 'case_0241', 'case_0242',...
                    'case_0248', 'case_0252', 'case_0253', 'case_0257', 'case_0258', 'Fdg_Stanford_001',...
                    'Fdg_Stanford_002', 'Fdg_Stanford_003', 'Fdg_Stanford_004', 'Fdg_Stanford_005',...
                    'Fdg_Stanford_006', 'Fdg_Stanford_010', 'Fdg_Stanford_011', 'Fdg_Stanford_012',...
                    'Fdg_Stanford_014', 'Fdg_Stanford_015', 'Fdg_Stanford_016', 'Fdg_Stanford_017',...
                    'Fdg_Stanford_023', 'Fdg_Stanford_024', 'Fdg_Stanford_025', 'Fdg_Stanford_026',...
                    'Fdg_Stanford_027', 'Fdg_Stanford_028', 'Fdg_Stanford_029', 'Fdg_Stanford_030',...
                    'Fdg_Stanford_031', 'Fdg_Stanford_032', '852_06182015', '869_06252015', '1496_05272016',...
                    '1498_05312016', '1512_06062016', '1549_06232016', '1559_06282016', '1572_07052016',...
                    '1604_07142016', '1610_07152016', '1619_07192016', '1657_08022016', '2002_02142017',...
                    '2010_02212017', '2114_04102017', '2120_04122017', '2142_04242017', '2275_07062017',...
                    '2277_07072017', '2284_07112017', '2292_07122017', '2295_07132017', '2310_07242017',...
                    '2374_08242017', '2377_08252017', '2388_08302017'};
test_set2 = {'Fdg_Stanford_026',...
                    'Fdg_Stanford_027', 'Fdg_Stanford_028', 'Fdg_Stanford_029', 'Fdg_Stanford_030',...
                    'Fdg_Stanford_031', 'Fdg_Stanford_032', '852_06182015', '869_06252015', '1496_05272016',...
                    '1498_05312016', '1512_06062016', '1549_06232016', '1559_06282016', '1572_07052016',...
                    '1604_07142016', '1610_07152016', '1619_07192016', '1657_08022016', '2002_02142017',...
                    '2010_02212017', '2114_04102017', '2120_04122017', '2142_04242017', '2275_07062017',...
                    '2277_07072017', '2284_07112017', '2292_07122017', '2295_07132017', '2310_07242017',...
                    '2374_08242017', '2377_08252017', '2388_08302017'};
test_set = {'2292_07122017', '2310_07242017', '1619_07192016', '869_06252015', 'Fdg_Stanford_032', ...
              '1604_07142016', '1498_05312016', 'Fdg_Stanford_027', '2388_08302017', '2010_02212017',...
            'Fdg_Stanford_026', 'Fdg_Stanford_029', '1610_07152016', '2142_04242017', '2295_07132017',...
            '2277_07072017', 'Fdg_Stanford_027','1559_06282016', '2002_02142017', '2284_07112017',...
            '2374_08242017','Fdg_Stanford_031'};

%test_set = {'case_0258', 'Fdg_Stanford_001'};

% test_set = [1375];
% test_set = [1355, 1732, 1947, 2516, 2063, 50767, 1375, 1758, 1923, 2425];
nifti_path = '/data/jiahong/project_zerodose_pytorch/src/nifti_0813/';
% save_path = '/data/jiahong/project_zerodose_pytorch/dicom/2020_12_16/';
% save_path = '/data/jiahong/project_zerodose_pytorch/dicom/2021_7_28/';
save_path = '/data/jiahong/project_zerodose_pytorch/dicom/2021_8_13/';
mkdir(save_path);
ref_path = '/data/jiahong/data/TPM.nii';
fulldose_path = '/data/jiahong/data/FDG_PET_selected_all/';
count = 0
gt_list = [];
pred_list = [];
subj_list = [];
for subj_id = test_set
    i = subj_id{1}
    disp(i)
    if ~ isfile([nifti_path i '_pred.nii'])
        continue
    end
    count = count + 1
    tpm1 = dir([fulldose_path i '/PET/*.dcm']);
    tpm2 = {tpm1.name};
    if length(tpm1) == 0
      continue
    end
    fulldose_subj_path = [fulldose_path i '/PET/' tpm2{1}];
    fulldose_info = dicominfo(fulldose_subj_path);
    uid = fulldose_info.StudyInstanceUID;
    start_loc = fulldose_info.ImagePositionPatient;
    PatientName.FamilyName = ['Patient' num2str(count, '%03d')];
    PatientName.GivenName = ['Patient' num2str(count, '%03d')];
    SeriesDisc = '20210927';

    subj_list = [subj_list, i]

    % pred
    fname_fullfile = [nifti_path i '_pred.nii'];
    % SeriesNo = floor(1000*rand());
    % if rand() < 0.5
    %   SeriesNo = 998;
    %   SeriesNo_gt = 999;
    % else
    %   SeriesNo = 999;
    %   SeriesNo_gt = 998;
    % end
    SeriesNo = 20210927;
    SeriesNo_gt = 1000;
    pred_list = [pred_list, SeriesNo]
    save_subj_path = [save_path 'Patient' num2str(count, '%03d') '/' num2str(SeriesNo, '%04d') '/'];
    mkdir(save_subj_path);
    make_fake_dicom(save_subj_path, fname_fullfile, start_loc, PatientName, SeriesNo, SeriesDisc, uid);

    % gt
    fname_fullfile = [nifti_path i '_PET.nii'];
    % SeriesNo = floor(1000*rand());
    SeriesNo = SeriesNo_gt;
    gt_list = [gt_list, SeriesNo]
    save_subj_path = [save_path 'Patient' num2str(count, '%03d') '/' num2str(SeriesNo, '%04d') '/'];
    mkdir(save_subj_path);
    make_fake_dicom(save_subj_path, fname_fullfile, start_loc, PatientName, SeriesNo, SeriesDisc, uid);

    % T1
    fname_fullfile = [nifti_path i '_T1.nii'];
    SeriesNo = 1001;
    save_subj_path = [save_path 'Patient' num2str(count, '%03d') '/' num2str(SeriesNo, '%04d') '/'];
    mkdir(save_subj_path);
    make_fake_dicom(save_subj_path, fname_fullfile, start_loc, PatientName, SeriesNo, SeriesDisc, uid);

    % T1c
    fname_fullfile = [nifti_path i '_T1c.nii'];
    SeriesNo = 1002;
    save_subj_path = [save_path 'Patient' num2str(count, '%03d') '/' num2str(SeriesNo, '%04d') '/'];
    mkdir(save_subj_path);
    make_fake_dicom(save_subj_path, fname_fullfile, start_loc, PatientName, SeriesNo, SeriesDisc, uid);

    % T2-FLAIR
    fname_fullfile = [nifti_path i '_T2_FLAIR.nii'];
    SeriesNo = 1003;
    save_subj_path = [save_path 'Patient' num2str(count, '%03d') '/' num2str(SeriesNo, '%04d') '/'];
    mkdir(save_subj_path);
    make_fake_dicom(save_subj_path, fname_fullfile, start_loc, PatientName, SeriesNo, SeriesDisc, uid);

    % ASL
    fname_fullfile = [nifti_path i '_ASL.nii'];
    SeriesNo = 1004;
    save_subj_path = [save_path 'Patient' num2str(count, '%03d') '/' num2str(SeriesNo, '%04d') '/'];
    mkdir(save_subj_path);
    make_fake_dicom(save_subj_path, fname_fullfile, start_loc, PatientName, SeriesNo, SeriesDisc, uid);

    disp(['finish' i])
end
