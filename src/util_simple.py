import os
import time
import pdb
# from glob import glob
import torch
# import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.misc as sci
import pickle
import shutil
import skimage
import skimage.io
import skimage.transform
import skimage.color
from skimage.measure import compare_nrmse, compare_psnr, compare_ssim
import skimage.metrics
import sklearn.metrics
import matplotlib as mpl
import nibabel as nib
import h5py
import pandas as pd
import nonechucks as nc
import yaml
import copy


def save_checkpoint(state, is_best, checkpoint_dir):
    print("save checkpoint")
    filename = checkpoint_dir+'/epoch'+str(state['epoch']).zfill(3)+'.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, checkpoint_dir+'/model_best.pth.tar')

'''
def load_checkpoint_by_key(values, checkpoint_dir, keys, device):
    '''
    the key can be state_dict for both optimizer or model,
    value is the optimizer or model that define outside
    '''
    filename = checkpoint_dir+'/model_best.pth.tar'
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        epoch = checkpoint['epoch']
        for i, key in enumerate(keys):
            values[i].load_state_dict(checkpoint[key])
        print("loaded checkpoint from '{}' (epoch: {}, monitor loss: {})".format(filename, \
                epoch, checkpoint['monitor_loss']))
    else:
        raise ValueError('No correct checkpoint')
    return values, epoch


def save_test_result(res, test_dir, bs, iter, save_att_maps=False, normalize_range=True, task='reconstruction'):
    '''self define function to save results or visualization'''
    sample_num = min(res['fake_B'].shape[0], bs)
    img_size = res['fake_B'].shape[2:]
    cm_jet = mpl.cm.get_cmap('jet')

    alpha_keys = [key for key in res.keys() if key.startswith('alpha')]
    alpha_keys.sort()
    for i in range(sample_num):
        idx = bs * iter + i
        # pdb.set_trace()
        # load and save input, output, target

        real_A = np.rot90(res['real_A'][i], axes=(1,2))
        imgs = []
        # pdb.set_trace()
        for j in range(real_A.shape[0]):
            maxA = np.max(real_A[j])
            if maxA == 0:
                real_A_j = real_A[j]
            else:
                real_A_j = real_A[j] / maxA
            imgs.append(real_A_j)

        real_B = np.rot90(res['real_B'][i].squeeze(0), axes=(0,1))
        fake_B = np.rot90(res['fake_B'][i].squeeze(0), axes=(0,1))

        if task == 'reconstruction':
            if not normalize_range: # lowdose
                stat = compute_stat(real_B, fake_B, task)

            maxB = np.max(real_B)
            if maxB > 0:
                real_B = real_B / maxB
            maxB = np.max(fake_B)
            if maxB > 0:
                fake_B = fake_B / maxB

            if normalize_range: # zerodose
                stat = compute_stat(real_B, fake_B, task)

        else:
            stat = None
            fake_B[fake_B>=0.5] = 1
            fake_B[fake_B<0.5] = 0

        imgs.append(real_B)
        imgs.append(fake_B)
        imgs.append(np.abs(real_B - fake_B))
        imgs = np.concatenate(imgs, axis=1)
        path = test_dir + '/' + str(idx).zfill(3) + '.jpg'
        sci.imsave(path, imgs)

        # save attention maps
        if save_att_maps and len(alpha_keys) > 0:
            img_size_rot = (img_size[1], img_size[0])
            att_maps = np.zeros((img_size[1], img_size[0]*len(alpha_keys)))
            for k, key in enumerate(alpha_keys):
                att_map = np.rot90(res[key][i].squeeze(0), axes=(0,1))
                # att_map = att_map / np.max(att_map)
                att_map = skimage.transform.resize(att_map, img_size_rot, preserve_range=True)
                att_maps[:, img_size_rot[1]*k:img_size_rot[1]*(k+1)] = att_map
            # att_maps = np.concatenate([att_maps, att_maps[:,img_size[0]*2:img_size[0]*3]], axis=1)
            att_maps = np.concatenate([att_maps, att_maps], axis=0)

            if task == 'reconstruction':
                background = real_B
                background_intensity = 30
            else:
                background = real_A[2]
                background_intensity = 0
            background_tile = np.tile(background+background_intensity, (1,len(alpha_keys)))
            background_tile = np.concatenate([background_tile, np.ones(background_tile.shape)], axis=0)
            background_hsv = skimage.color.rgb2hsv(np.dstack((background_tile, background_tile, background_tile)))
            att_maps_rgba = cm_jet(att_maps)
            att_maps_hsv = skimage.color.rgb2hsv(att_maps_rgba[:,:,:3])
            background_hsv[..., 0] = att_maps_hsv[..., 0]
            background_hsv[..., 1] = att_maps_hsv[..., 1] * 0.5
            fusion = skimage.color.hsv2rgb(background_hsv)
            path = test_dir + '/' + str(idx).zfill(3) + '_att_maps.jpg'
            sci.imsave(path, fusion)

    return stat

def save_test_result_by_volume(save_dict_list, test_dir, save_nifti=True, task='reconstruction', slice_per_subj=115):
    # reorganize save_dict_list
    save_dict_new = {}
    for key in ['real_B', 'fake_B']:
        save_dict_new[key] = np.concatenate([save_dict_list[i][key] for i in range(len(save_dict_list))], axis=0).squeeze(axis=1)

    slice_num = save_dict_new['real_B'].shape[0]
    subj_num = slice_num / slice_per_subj
    if subj_num * slice_per_subj != slice_num:
        print('Might missing some slices!')

    if task == 'reconstruction':
        stat_list_volume = {'psnr':[], 'ssim':[], 'rmse':[]}
    else:
        stat_list_volume = {'auc':[], 'dice':[], 'tpr':[], 'tnr':[], 'alvd':[]}
    for i in range(subj_num):
        real_B = save_dict_new['real_B'][slice_per_subj*i:slice_per_subj*(i+1),:,:]
        fake_B = save_dict_new['fake_B'][slice_per_subj*i:slice_per_subj*(i+1),:,:]

        # save volumes to nifti
        if save_nifti:
            path = test_dir + '/subj_' + str(i) + '_real.nii'
            save_volume_nifti(path, real_B)
            path = test_dir + '/subj_' + str(i) + '_fake.nii'
            save_volume_nifti(path, fake_B)

        # compute metrics
        if task == 'reconstruction':
            stat_list= {'psnr':[], 'ssim':[], 'rmse':[]}
            real_B = real_B / real_B.max()
            fake_B = fake_B / fake_B.max()
            for j in range(real_B.shape[0]):
                stat = compute_stat(real_B[j,:,:], fake_B[j,:,:], task)
                for key in stat.keys():
                    stat_list[key].append(stat[key])
        else:
            # stat_list = {'auc':[], 'dice':[], 'tpr':[], 'tnr':[], 'alvd':[]}
            stat_list = compute_stat(real_B, fake_B, task)
        # compute mean for each volume
        for key in stat_list.keys():
            stat = np.array(stat_list[key])
            stat_mean = stat.mean()
            stat_var = stat.var()
            stat_list_volume[key].append(stat_mean)
            print('volume {} {} mean:{}, var:{}'.format(i, key, stat_mean, stat_var))

    return stat_list_volume

def save_volume_nifti(save_path, data):
    # transpose [c, h, w] to [h, w, c]
    data = np.transpose(data, (1,2,0))
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, save_path)

def compute_stat(real_B, fake_B, task='reconstruction'):
    if task == 'reconstruction':
        range = np.max(real_B)
        # if range < 1:   # for range between 0~1
        #     range = 1
        try:
            rmse_pred = compare_nrmse(real_B, fake_B)
        except:
            rmse_pred = float('nan')
        try:
            psnr_pred = compare_psnr(real_B, fake_B)
            # psnr_pred = compare_psnr(real_B, fake_B, data_range=range)
        except:
            psnr_pred = float('nan')
        try:
            ssim_pred = compare_ssim(real_B, fake_B)
            # ssim_pred = compare_ssim(real_B, fake_B, data_range=range)
        except:
            ssim_pred = float('nan')
        return {'psnr':psnr_pred, 'ssim':ssim_pred, 'rmse':rmse_pred}
    else:
        # AUC:
        thres = 0.5
        # pdb.set_trace()
        fake_B[fake_B>=thres] = 1
        fake_B[fake_B<thres] = 0
        real_B = real_B.flatten()
        fake_B = fake_B.flatten()
        try:
            auc = sklearn.metrics.roc_auc_score(real_B, fake_B)
        except:
            auc = float('nan')
        metrics = classification_metrics(real_B, fake_B)
        alvd = np.abs(real_B.sum() - fake_B.sum())
        return {'auc':auc, 'dice':metrics['dice'], 'tpr':metrics['tpr'],
                'tnr':metrics['tnr'], 'alvd':alvd}

def classification_metrics(real_B, fake_B):
    all_num = real_B.shape[0]
    true_pos = ((fake_B==1.) & (real_B==1.)).sum()
    true_neg = ((fake_B==0.) & (real_B==0.)).sum()
    false_pos = ((fake_B==1.) & (real_B==0.)).sum()
    false_neg = ((fake_B==0.) & (real_B==1.)).sum()

    # sensitivity, recall, hit rate, or true positive rate
    tpr = true_pos / float(true_pos + false_neg)
    # specificity, selectivity or true negative rate
    tnr = true_neg / float(true_neg + false_pos)
    # precision or positive predictive value
    ppv = true_pos / float(true_pos + false_pos)
    # negative predictive value
    npv = true_neg / float(true_neg + false_neg)
    # miss rate or false negative rate
    fnr = 1 - tpr
    # fall-out or false positive rate
    fpr = 1 - tnr
    # false discovery rate
    fdr = 1 - ppv
    # false omission rate
    fomr = 1 - npv
    # accuracy
    acc = (true_pos + true_neg) / float(true_pos + true_neg + false_pos + false_neg)
    # dice, f1
    dice = 2*true_pos / float(2*true_pos + false_pos + false_neg)
    # iou, jaccard
    iou = true_pos / float(true_pos + false_pos + false_neg)

    return {'tpr':tpr, 'tnr':tnr, 'ppv':ppv, 'npv':npv, 'fnr':fnr, 'fpr':fpr,
            'fdr':fdr, 'fomr':fomr, 'acc':acc, 'dice':dice, 'iou':iou}



def save_test_classification_result(label_list, prediction_list, save_path):
    '''self define function to save classification results'''

    label_list = np.array([item for sublist in label_list for item in sublist])
    prediction_list = np.array([item for sublist in prediction_list for item in sublist])
    majority_vote_volume_prediction(prediction_list, label_list)
    output = np.vstack([label_list, prediction_list.squeeze(-1)]).T
    np.save(save_path, output)

    return np.mean(np.abs(label_list - prediction_list))

def majority_vote_volume_prediction(prediction_list, label_list, slice_per_subj=48):
    subj_num = prediction_list.shape[0] / slice_per_subj
    pred_vol_list = []
    label_vol_list = []
    for subj in range(subj_num):
        data = prediction_list[subj*slice_per_subj+10:(subj+1)*slice_per_subj-10]
        pred_vol_list.append(data.mean())
        data = label_list[subj*slice_per_subj:(subj+1)*slice_per_subj]
        label_vol_list.append(data.mean())
    print(pred_vol_list)
    print(label_vol_list)

# make the pickle data to test the amyloid classifier on the generated images
def save_dir_jpg_to_pickle(img_list, pkl_path):
    sample_list = []
    img_list = np.transpose(img_list, [0,2,3,1])
    for i, img in enumerate(img_list):
        sample = {'target': img}
        sample_list.append(sample)
    with open(pkl_path, 'wb') as handle:
        pickle.dump(sample_list, handle, protocol=2)
    print('save output images to pickle file!')


def save_args_info(args):
    path = args.args_dir + '/' + args.task_label + '_' + args.phase + '.txt'
    txt_file = open(path, 'w')
    args_dict = vars(args)
    for key in args_dict.keys():
        value = args_dict[key]
        txt_file.write(key+' : '+str(value)+'\n')
    txt_file.close()

def save_stat_to_args(args, stats_dict={}):
    path = args.args_dir + '/' + args.task_label + '_test' + '.txt'
    txt_file = open(path, 'a')
    for key in stats_dict.keys():
        value = stats_dict[key]
        txt_file.write(key+' : '+str(value)+'\n')
    txt_file.close()

def save_stat_to_args_volume(args, stats_dict={}):
    path = args.args_dir + '/' + args.task_label + '_test' + '.txt'
    txt_file = open(path, 'a')
    txt_file.write('volume-wise statistics \n')
    for key in stats_dict.keys():
        value = stats_dict[key]
        txt_file.write(key+' : '+str(value)+'\n')
    txt_file.close()

'''

# new code, 2020/08/19
class ZeroDoseDataset(Dataset):
    def __init__(self, dataset_name, data, subj_list, idx_list, block_size=3, contrast_list=['T1'], aug=False, dropoff=False):
        self.dataset_name = dataset_name
        self.data = data
        # self.subj_list = subj_list.astype(np.string_)
        self.subj_list = subj_list
        self.idx_list = idx_list
        # pdb.set_trace()
        self.contrast_name = contrast_list
        self.block_size = block_size
        self.contrast_list = contrast_list
        self.aug = aug
        self.dropoff = dropoff
        # self.image_size = self.data[self.subj_list[0]+'/PET'][:,:,0].shape
        # self.image_size = [160, 192]
        self.image_size = [256, 256]

    def __len__(self):
        return len(self.subj_list)

    def __getitem__(self, idx):
        try:
            # pdb.set_trace()
            subj_id = str(self.subj_list[idx])
            slice_idx = self.idx_list[idx]

            if slice_idx < self.block_size:
                slice_idx = self.block_size
            if self.dataset_name == 'Tau' or 'rawspace' in self.dataset_name:
                if  slice_idx > 88 - self.block_size:
                    slice_idx = 88 - self.block_size
            else:
                if  slice_idx > 155 - self.block_size:
                    slice_idx = 155 - self.block_size

            imgs = []
            mask = []
            drop_num = 0

            for contrast_name in self.contrast_list:
                # normal
                if contrast_name == "T1c" or contrast_name == "T1":
                    if subj_id+'/'+contrast_name+'_GRE' in self.data.keys() and subj_id+'/'+contrast_name+'_SE' in self.data.keys():
                        if self.aug:
                            if np.random.rand() > 0.5:
                                imgs.append(self.data[subj_id+'/'+contrast_name+'_GRE'][:,:,slice_idx-self.block_size:slice_idx+self.block_size+1])
                            else:
                                imgs.append(self.data[subj_id+'/'+contrast_name+'_SE'][:,:,slice_idx-self.block_size:slice_idx+self.block_size+1])
                        else:
                            imgs.append(self.data[subj_id+'/'+contrast_name+'_GRE'][:,:,slice_idx-self.block_size:slice_idx+self.block_size+1])
                        mask.append(1)
                    elif subj_id+'/'+contrast_name+'_GRE' in self.data.keys():
                        imgs.append(self.data[subj_id+'/'+contrast_name+'_GRE'][:,:,slice_idx-self.block_size:slice_idx+self.block_size+1])
                        mask.append(1)
                    elif subj_id+'/'+contrast_name+'_SE' in self.data.keys():
                        imgs.append(self.data[subj_id+'/'+contrast_name+'_SE'][:,:,slice_idx-self.block_size:slice_idx+self.block_size+1])
                        mask.append(1)
                    else:
                        imgs.append(np.zeros((self.image_size[0], self.image_size[1], 2*self.block_size+1)))
                        mask.append(0)
                else:
                    if subj_id+'/'+contrast_name in self.data:
                        imgs.append(self.data[subj_id+'/'+contrast_name][:,:,slice_idx-self.block_size:slice_idx+self.block_size+1])
                        mask.append(1)
                    else:
                        # print(subj_id+'/'+contrast_name)
                        imgs.append(np.zeros((self.image_size[0], self.image_size[1], 2*self.block_size+1)))
                        mask.append(0)

            mask = np.array(mask)
            inputs = np.concatenate(imgs, 2)

            if self.dataset_name == 'ZeroDose' or self.dataset_name == 'ZeroDose_rawspace':
                if subj_id+'/PET' in self.data.keys():
                    targets = self.data[subj_id+'/PET'][:,:,slice_idx:slice_idx+1]
                elif subj_id+'/PET_MAC' in self.data.keys():
                    targets = self.data[subj_id+'/PET_MAC'][:,:,slice_idx:slice_idx+1]
                elif subj_id+'/PET_QCLEAR' in self.data.keys():
                    targets = self.data[subj_id+'/PET_QCLEAR'][:,:,slice_idx:slice_idx+1]
                else:
                    targets = self.data[subj_id+'/PET_TOF'][:,:,slice_idx:slice_idx+1]
            else:
                raise ValueError('Not support other dataset yet!')

            if self.dropoff and mask.sum() > 1:
                if np.random.rand() > 0.8:
                    drop_idx = np.random.choice(np.where(mask==1)[0], 1)[0]
                    inputs[:,:,drop_idx*(2*self.block_size+1):(drop_idx+1)*(2*self.block_size+1)] = 0
                    mask[drop_idx] = 0

            inputs = np.transpose(inputs, (2, 0, 1))
            targets = np.transpose(targets, (2, 0, 1))

            # if self.aug:
            #     if np.random.rand() > 0.5:
            #         inputs = inputs[:,::-1] - np.zeros_like(inputs)
            #         targets = targets[::-1] - np.zeros_like(targets)

            mask_img = (inputs[0] == 0).astype(float)

            return {'inputs': inputs, 'targets': targets, 'subj_id': subj_id, 'slice_idx': slice_idx, 'mask': mask, 'mask_img': mask_img}
        except:
            return None


class ZeroDoseDataAll(object):
    def __init__(self, dataset_name, data_path, data_h5_path, train_txt_path, val_txt_path, test_txt_path,
            batch_size=16, shuffle=True, num_workers=0, block_size=3, contrast_list=['T1'], aug=False, dropoff=False, skull_strip=False):
        if dataset_name == 'ZeroDose' or dataset_name == 'ZeroDose_rawspace':

            data = h5py.File(data_h5_path, 'r')
            subj_list_train, idx_list_train = self.load_idx_list(train_txt_path)
            subj_list_val, idx_list_val = self.load_idx_list(val_txt_path)
            subj_list_test, idx_list_test = self.load_idx_list(test_txt_path)
        else:
            raise ValueError('Not supporting other dataset yet!')

        brain_mask_nib = nib.load(os.path.join(data_path, 'tpm_mask_new.nii'))
        brain_mask = brain_mask_nib.get_fdata()

        train_dataset = nc.SafeDataset(ZeroDoseDataset(dataset_name, data, subj_list_train, idx_list_train, block_size=block_size, contrast_list=contrast_list, aug=aug, dropoff=dropoff))
        val_dataset = nc.SafeDataset(ZeroDoseDataset(dataset_name, data, subj_list_val, idx_list_val, block_size=block_size, contrast_list=contrast_list, aug=False, dropoff=dropoff))
        test_dataset = nc.SafeDataset(ZeroDoseDataset(dataset_name, data, subj_list_test, idx_list_test, block_size=block_size, contrast_list=contrast_list, aug=False, dropoff=False))

        self.trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.valLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.testLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def load_idx_list(self, file_path):
        lines = pd.read_csv(file_path, sep=" ", header=None)
        return np.array(lines.iloc[:,0]), np.array(lines.iloc[:,1])

'''
class ZeroDoseDataset3D(Dataset):
    def __init__(self, dataset_name, data, subj_list, contrast_list=['T1'], aug=False, missing=False, dropoff=False):
        self.dataset_name = dataset_name
        self.data = data
        self.subj_list = subj_list
        self.contrast_name = contrast_list
        self.contrast_list = contrast_list
        self.aug = aug
        self.missing = missing
        self.dropoff = dropoff
        self.image_size = [160, 192, 64]

    def __len__(self):
        return len(self.subj_list)

    def __getitem__(self, idx):
        try:
            subj_id = self.subj_list[idx]
            imgs = []
            mask = []
            drop_num = 0
            for contrast_name in self.contrast_list:
                # normal
                if subj_id+'/'+contrast_name in self.data:
                    if self.dataset_name == 'ZeroDose':
                        imgs.append(self.data[subj_id+'/'+contrast_name][:,:,45:-47])
                    else:
                        imgs.append(self.data[subj_id+'/'+contrast_name][:,:,45:-46])
                    mask.append(1)
                else:
                    imgs.append(np.zeros((self.image_size[0], self.image_size[1], self.image_size[2])))
                    mask.append(0)

            mask = np.array(mask)
            inputs = np.stack(imgs, 0)
            if self.dataset_name == 'ZeroDose':
                if subj_id+'/PET' in self.data.keys():
                    targets = self.data[subj_id+'/PET'][:,:,45:-47]
                else:
                    targets = np.zeros((self.image_size[0], self.image_size[1], self.image_size[2]))
            elif self.dataset_name == 'BraTS':
                if subj_id+'/seg' in self.data.keys():
                    targets = self.data[subj_id+'/seg'][:,:,45:-46]
                    #### if segmenting all potential tumors
                    # targets[targets>0] = 1.
                    #### if segmenting only Gd enhancing tumors
                    targets[targets<4] = 0.
                    targets[targets==4] = 1.
                else:
                    targets = np.zeros((self.image_size[0], self.image_size[1], self.image_size[2]))
            else:
                targets = np.zeros((self.image_size[0], self.image_size[1], self.image_size[2]))

            # if self.dropoff and mask.sum() > 1:
            #     if np.random.rand() > 0.8:
            #         drop_idx = np.random.choice(np.where(mask==1)[0], 1)[0]
            #         inputs[drop_idx] = 0
            #         mask[drop_idx] = 0
            # # print(inputs.shape, targets.shape)
            #
            # if self.aug:
            #     if np.random.rand() > 0.5:
            #         inputs = inputs[:,::-1] - np.zeros_like(inputs)
            #         targets = targets[::-1] - np.zeros_like(targets)
            #     rand_scale = 1 + 0.2 * (np.random.rand() - 0.5)
            #     rand_shift = 0.2 * (np.random.rand() - 0.5)
            #     inputs = inputs * rand_scale + rand_shift
            #     inputs[inputs == inputs.min()] = -10

            return {'inputs': inputs, 'targets': targets, 'subj_id': subj_id, 'mask': mask, 'slice_idx': 0}
        except:
            print('Failed', idx)
            return None

class ZeroDoseDataAll3D(object):
    def __init__(self, dataset_name, data_path, norm_type='mean', batch_size=16, num_fold=5, fold=0, shuffle=True, num_workers=0, block_size=3, contrast_list=['T1'], aug=False, dropoff=False, skull_strip=False):
        if dataset_name == 'BraTS':
            if norm_type == 'mean':
                data = h5py.File(os.path.join(data_path, 'BraTS_All.h5'), 'r')
            else:
                data = h5py.File(os.path.join(data_path, 'BraTS_All_zscore_10.h5'), 'r')
            subj_list_train = self.load_idx_list(os.path.join(data_path, 'fold_BraTS_3d_'+str(fold)+'_train_noval.txt'))
            subj_list_val = self.load_idx_list(os.path.join(data_path, 'fold_BraTS_3d_'+str(fold)+'_val_noval.txt'))
            subj_list_test = self.load_idx_list(os.path.join(data_path, 'fold_BraTS_3d_'+str(fold)+'_test_noval.txt'))
        elif dataset_name == 'ZeroDose':
            if norm_type == 'mean':
                data = h5py.File(os.path.join(data_path, 'tumor_complete_mean.h5'), 'r')
            else:
                data = h5py.File(os.path.join(data_path, 'tumor_complete_zscore.h5'), 'r')
            subj_list_train = self.load_idx_list(os.path.join(data_path, 'tumor_complete_3d_all.txt'))
            subj_list_val = self.load_idx_list(os.path.join(data_path, 'tumor_complete_3d_all.txt'))
            subj_list_test = self.load_idx_list(os.path.join(data_path, 'tumor_complete_3d_all.txt'))
        # else:
            # raise ValueError('Did not implement 3D dataset other than BraTS')

        train_dataset = ZeroDoseDataset3D(dataset_name, data, subj_list_train, contrast_list=contrast_list, aug=False, dropoff=dropoff)
        val_dataset = ZeroDoseDataset3D(dataset_name, data, subj_list_val, contrast_list=contrast_list, aug=False, dropoff=dropoff)
        test_dataset = ZeroDoseDataset3D(dataset_name, data, subj_list_test, contrast_list=contrast_list, aug=False, dropoff=False)

        self.trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.valLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.testLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def load_idx_list(self, file_path):
        lines = pd.read_csv(file_path, sep=" ")
        return np.array(lines.iloc[:,0])

'''

# save config values
def save_config_file(config):
    file_path = os.path.join(config['ckpt_path'], 'config.txt')
    f = open(file_path, 'w')
    for key, value in config.items():
        f.write(key + ': ' + str(value) + '\n')
    f.close()

# save results statistics
def save_result_stat(stat, config, info='Default'):
    stat_path = os.path.join(config['ckpt_path'], 'stat.csv')
    columns=['info',] + sorted(stat.keys())
    if not os.path.exists(stat_path):
        df = pd.DataFrame(columns=columns)
        df.to_csv(stat_path, mode='a', header=True)

    stat['info'] = info
    for key, value in stat.items():
        stat[key] = [value]
    df = pd.DataFrame.from_dict(stat)
    df = df[columns]
    df.to_csv(stat_path, mode='a', header=False)


# load model
def load_checkpoint_by_key(values, checkpoint_dir, keys, device, ckpt_name='model_best.pth.tar'):
    '''
    the key can be state_dict for both optimizer or model,
    value is the optimizer or model that define outside
    '''
    filename = os.path.join(checkpoint_dir, ckpt_name)
    print(filename)
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        epoch = checkpoint['epoch']
        for i, key in enumerate(keys):
            try:
                if key == 'model':
                    values[i] = load_checkpoint_model(values[i], checkpoint[key])
                else:
                    values[i].load_state_dict(checkpoint[key])
                print('loading ' + key + ' success!')
            except:
                print('loading ' + key + ' failed!')
        print("loaded checkpoint from '{}' (epoch: {}, monitor metric: {})".format(filename, \
                epoch, checkpoint['monitor_metric']))
    else:
        raise ValueError('No correct checkpoint')
    return values, epoch

def load_checkpoint_model(model, pretrained_dict):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape==model_dict[k].shape}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model

def load_config_yaml(yaml_path):
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        return True, config
    else:
        return False, None

def save_config_yaml(ckpt_path, config):
    yaml_path = os.path.join(ckpt_path, 'config.yaml')
    remove_key = []
    for key in config.keys():
        if isinstance(config[key], int) or isinstance(config[key], float) or isinstance(config[key], str) or isinstance(config[key], list)  or isinstance(config[key], dict):
            continue
        remove_key.append(key)
    config_copy = copy.deepcopy(config)
    for key in remove_key:
        config_copy.pop(key, None)
    with open(yaml_path, 'w') as file:
        documents = yaml.dump(config_copy, file)
    print('Saved yaml file')


def compute_recon_loss(gt, output, p=2):
    # loss on reconstruction
    if p == 1:
        return torch.abs(gt - output).mean()
    else:
        return torch.pow(gt - output, 2).mean()

def compute_reconstruction_metrics(target, pred):
    ssim_list = []
    rmse_list = []
    psnr_list = []
    for i in range(target.shape[0]):
        metrics_dict = compute_reconstruction_metrics_single(target[i,0], pred[i,0])
        ssim_list.append(metrics_dict['ssim'])
        psnr_list.append(metrics_dict['psnr'])
        rmse_list.append(metrics_dict['rmse'])
    return {'ssim': ssim_list, 'psnr': psnr_list, 'rmse': rmse_list}


def compute_reconstruction_metrics_single(target, pred):
    # target = target / target.max() + 1e-8
    # pred = pred / pred.max() + 1e-8
    # range = np.max(target) - np.min(target)
    # pdb.set_trace()
    target = target - target.min()
    pred = pred - pred.min()
    range = target.max()
    try:
        rmse_pred = skimage.metrics.mean_squared_error(target, pred)
        # rmse_pred = skimage.metrics.normalized_root_mse(target, pred)
    except:
        rmse_pred = float('nan')
    try:
        # psnr_pred = skimage.metrics.peak_signal_noise_ratio(target, pred)
        psnr_pred = skimage.metrics.peak_signal_noise_ratio(target, pred, data_range=range)
    except:
        psnr_pred = float('nan')
    try:
        # ssim_pred = skimage.metrics.structural_similarity(target, pred)
        ssim_pred = skimage.metrics.structural_similarity(target, pred, data_range=range)
    except:
        ssim_pred = float('nan')
    return {'ssim': ssim_pred, 'rmse': rmse_pred, 'psnr': psnr_pred}

'''
def compute_segmentation_metrics(target, pred):
    dice_list = []
    iou_list = []
    for i in range(target.shape[0]):
        metrics_dict = compute_segmentation_metrics_single(target[i], pred[i])
        dice_list.append(metrics_dict['dice'])
        iou_list.append(metrics_dict['iou'])
    return {'dice': dice_list, 'iou': iou_list}

def compute_segmentation_metrics_single(target, pred):
    if target.shape[0] == 1:
        target = target.squeeze(0)
    intersection = np.logical_and((target == 1), (pred > 0.5))
    union = np.logical_or((target == 1), (pred > 0.5))
    dice = (2. * intersection.sum() + 1) / ((target == 1).sum() + (pred > 0.5).sum() + 1)
    iou = (np.sum(intersection) + 1) / (np.sum(union) + 1)

    return {'dice': dice, 'iou': iou}
'''