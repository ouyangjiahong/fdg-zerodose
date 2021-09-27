# new version, 2020/08/19

import os
import glob
import time
import torch
import torch.optim as optim
import numpy as np
import yaml
import pdb
import psutil

from model import *
from util import *

# set seed
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True


_, config = load_config_yaml('config.yaml')
config['in_num_ch'] = len(config['contrast_list']) * (2*config['block_size']+1)
config['device'] = torch.device('cuda:'+ config['gpu'])

if config['ckpt_timelabel'] and (config['phase'] == 'test' or config['continue_train'] == True):
    time_label = config['ckpt_timelabel']
else:
    localtime = time.localtime(time.time())
    time_label = str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min)


# ckpt folder, load yaml config
# config['ckpt_path'] = os.path.join('../ckpt/', 'BraTS', config['model_name'], time_label)
config['ckpt_path'] = os.path.join('../ckpt/', config['dataset_name'], config['model_name'], time_label)
print(config['ckpt_path'])

if not os.path.exists(config['ckpt_path']):     # test, not exists
    os.makedirs(config['ckpt_path'])
    save_config_yaml(config['ckpt_path'], config)
elif config['load_yaml']:       # exist and use yaml config
    flag, config_load = load_config_yaml(os.path.join(config['ckpt_path'], 'config.yaml'))
    if flag:    # load yaml success
        print('load yaml config file')
        for key in config_load.keys():  # if yaml has, use yaml's param, else use config
            if key == 'phase':
                continue
            if key in config.keys():
                config[key] = config_load[key]
            else:
                print('current config do not have yaml param')
        config['in_num_ch'] = len(config['contrast_list']) * (2*config['block_size']+1)
    else:
        save_config_yaml(config['ckpt_path'], config)

print(config['model_name'])
# config['ckpt_name'] = 'model_best.pth.tar'

Data = ZeroDoseDataAll(config['dataset_name'], config['data_path'], norm_type=config['norm_type'], batch_size=config['batch_size'], num_fold=config['num_fold'], \
                        fold=config['fold'], shuffle=config['shuffle'], num_workers=0, block_size=config['block_size'], \
                        contrast_list=config['contrast_list'], aug=False, dropoff=config['dropoff'], skull_strip=config['skull_strip'])
trainDataLoader = Data.trainLoader
valDataLoader = Data.valLoader
testDataLoader = Data.testLoader

# config['dataset_name'] = 'BraTS'

# define model
if config['model_name'] == 'Simple':
    if config['norm_type'] == 'z-score':
        config['target_output_act'] = 'no'
    else:
        config['target_output_act'] = 'softplus'

    if config['target_model_name'] == 'U':
        model = GANShortGenerator(in_num_ch=config['in_num_ch'], out_num_ch=config['out_num_ch'], first_num_ch=64, input_size=(config['input_height'], config['input_width']), output_activation=config['target_output_act']).to(config['device'])
    elif config['target_model_name'] == 'U+SA':
        model = GANShortGeneratorWithSpatialAttention(in_num_ch=config['in_num_ch'], out_num_ch=config['out_num_ch'], first_num_ch=64, input_size=(config['input_height'], config['input_width']), output_activation=config['target_output_act']).to(config['device'])
    elif config['target_model_name'] == 'U+SA+CA':
        model = GANShortGeneratorWithChannelAttentionAllAndSpatialAttention(in_num_ch=config['in_num_ch'], out_num_ch=config['out_num_ch'], first_num_ch=64, input_size=(config['input_height'], config['input_width']), output_activation=config['target_output_act']).to(config['device'])
    elif config['target_model_name'] == 'U+SSA+CA':
        model = GANShortGeneratorWithChannelAttentionAllAndSymmetrySpatialAttention(in_num_ch=config['in_num_ch'], out_num_ch=config['out_num_ch'], first_num_ch=64, input_size=(config['input_height'], config['input_width']), output_activation=config['target_output_act']).to(config['device'])
    else:
        raise ValueError('Not implemented')
else:
    raise ValueError('Not implemented')

class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, max_epoch, power=0.9, last_epoch=-1):
        self.max_epoch = max_epoch
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_epoch) ** self.power
                for base_lr in self.base_lrs]

# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5, amsgrad=True)
# scheduler = PolyLR(optimizer, max_epoch=300, power=0.9)
scheduler = PolyLR(optimizer, max_epoch=50, power=0.9)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-5)


# load pretrained model
if config['continue_train'] or config['phase'] == 'test':
    [optimizer, scheduler, model], start_epoch = load_checkpoint_by_key([optimizer, scheduler, model], config['ckpt_path'], ['optimizer', 'scheduler', 'model'], config['device'], config['ckpt_name'])
    # [optimizer, model], start_epoch = load_checkpoint_by_key([optimizer, model], config['ckpt_path'], ['optimizer', 'model'], config['device'], config['ckpt_name'])
    # [model], start_epoch = load_checkpoint_by_key([model], config['ckpt_path'], ['model'], config['device'], config['ckpt_name'])
else:
    start_epoch = -1

if config['phase'] == 'train':
    save_config_file(config)

# train
def train():
    global_iter = 0
    monitor_metric_best = 100
    start_time = time.time()

    # stat = evaluate(phase='val', set='val', save_res=False)
    # print(stat)
    for epoch in range(start_epoch+1, config['epochs']):
        model.train()
        loss_all_dict = {'recon_y': 0., 'all': 0.}
        global_iter0 = global_iter
        for iter, sample in enumerate(trainDataLoader, 0):
            global_iter += 1

            inputs = sample['inputs'].to(config['device'], dtype=torch.float)
            targets = sample['targets'].to(config['device'], dtype=torch.float)
            mask = sample['mask'].to(config['device'], dtype=torch.float)

            pred, _ = model(inputs)

            if config['p'] == 1:
                loss_recon_y = torch.abs(pred - targets).mean()
            else:
                loss_recon_y = torch.pow(pred - targets, 2).mean()

            loss = config['lambda_recon_y'] * loss_recon_y

            loss_all_dict['recon_y'] += loss_recon_y.item()
            loss_all_dict['all'] += loss.item()

            # pdb.set_trace()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for name, param in model.named_parameters():
                try:
                    if not torch.isfinite(param.grad).all():
                        pdb.set_trace()
                except:
                    continue

            optimizer.step()
            optimizer.zero_grad()

            if global_iter % 10 == 0:
                print('Epoch[%3d], iter[%3d]: loss=[%.4f], recon y=[%.4f]' \
                        % (epoch, iter, loss.item(), loss_recon_y.item()))

            # if iter > 3:
            #     break

        # save train result
        num_iter = global_iter - global_iter0
        for key in loss_all_dict.keys():
            loss_all_dict[key] /= num_iter
        save_result_stat(loss_all_dict, config, info='epoch[%2d]'%(epoch))
        print(loss_all_dict)

        # validation
        # pdb.set_trace()
        stat = evaluate(phase='val', set='val', save_res=False)
        monitor_metric = stat['recon_y']
        # scheduler.step(monitor_metric)
        scheduler.step()
        save_result_stat(stat, config, info='val')
        print(stat)

        # save ckp
        is_best = False
        if monitor_metric <= monitor_metric_best:
            is_best = True
            monitor_metric_best = monitor_metric if is_best == True else monitor_metric_best
        state = {'epoch': epoch, 'monitor_metric': monitor_metric, 'stat': stat, \
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), \
                'model': model.state_dict()}
        save_checkpoint(state, is_best, config['ckpt_path'])

def evaluate(phase='val', set='val', save_res=True, info=''):
    model.eval()
    if phase == 'val':
        loader = valDataLoader
    else:
        if set == 'train':
            loader = trainDataLoader
        elif set == 'val':
            loader = valDataLoader
        elif set == 'test':
            loader = testDataLoader
        else:
            raise ValueError('Undefined loader')

    loss_all_dict = {'recon_y': 0., 'all': 0.}

    subj_id_list = []
    slice_idx_list = []
    input_list = []
    target_list = []
    mask_list = []
    y_fake_fused_list = []
    att_map_list2 = []
    att_map_list3 = []
    metrics_list_dict = {}

    res_path = os.path.join(config['ckpt_path'], 'result_'+set)
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    with torch.no_grad():
        for iter, sample in enumerate(loader, 0):
            subj_id = sample['subj_id']
            inputs = sample['inputs'].to(config['device'], dtype=torch.float)
            targets = sample['targets'].to(config['device'], dtype=torch.float)
            mask = sample['mask'].to(config['device'], dtype=torch.float)
            slice_idx = sample['slice_idx'].to(config['device'], dtype=torch.float)

            pred, att_map_dict = model(inputs)

            if config['p'] == 1:
                loss_recon_y = torch.abs(pred - targets).mean()
            else:
                loss_recon_y = torch.pow(pred - targets, 2).mean()

            loss = config['lambda_recon_y'] * loss_recon_y

            loss_all_dict['recon_y'] += loss_recon_y.item()
            loss_all_dict['all'] += loss.item()

            metrics = compute_reconstruction_metrics(targets.detach().cpu().numpy(), pred.detach().cpu().numpy())

            print(metrics)

            for key in metrics.keys():
                if key in metrics_list_dict.keys():
                    metrics_list_dict[key].extend(metrics[key])
                else:
                    metrics_list_dict[key] = metrics[key]
            # print(metrics)
            # pdb.set_trace()

            if phase == 'test' and save_res:
                input_list.append(inputs.detach().cpu().numpy())
                target_list.append(targets.detach().cpu().numpy())
                y_fake_fused_list.append(pred.detach().cpu().numpy())
                subj_id_list.append(subj_id)
                slice_idx_list.append(slice_idx.detach().cpu().numpy())
                mask_list.append(mask.detach().cpu().numpy())
                if 'alpha_3' in att_map_dict:
                    att_map_list2.append(att_map_dict['alpha_2'].detach().cpu().numpy())
                    att_map_list3.append(att_map_dict['alpha_3'].detach().cpu().numpy())

            #
            # if iter > 3:
            #     break

    for key in loss_all_dict.keys():
        loss_all_dict[key] /= (iter + 1)

    for key in metrics_list_dict:
        loss_all_dict[key] = np.array(metrics_list_dict[key]).mean()

    if phase == 'test' and save_res:
        input_list = np.concatenate(input_list, axis=0)
        target_list = np.concatenate(target_list, axis=0)
        slice_idx_list = np.concatenate(slice_idx_list, axis=0)
        subj_id_list = np.concatenate(subj_id_list, axis=0)
        y_fake_fused_list = np.concatenate(y_fake_fused_list, axis=0)
        mask_list = np.concatenate(mask_list, axis=0)

        path = os.path.join(res_path, 'results_all'+info+'.h5')
        if os.path.exists(path):
            print('Already saved h5')
        else:
            h5_file = h5py.File(path, 'w')
            h5_file.create_dataset('subj_id', data=np.string_(subj_id_list))
            h5_file.create_dataset('slice_idx', data=slice_idx_list)
            h5_file.create_dataset('inputs', data=input_list)
            h5_file.create_dataset('targets', data=target_list)
            h5_file.create_dataset('mask', data=mask_list)
            h5_file.create_dataset('y_fake_fused', data=y_fake_fused_list)
            if len(att_map_list2) > 0:
                att_map_list2 = np.concatenate(att_map_list2, axis=0)
                att_map_list3 = np.concatenate(att_map_list3, axis=0)
                h5_file.create_dataset('att_map_2', data=att_map_list2)
                h5_file.create_dataset('att_map_3', data=att_map_list3)

        # save nifti
        nifti_path = os.path.join(res_path, 'nifti/')
        if not os.path.exists(nifti_path):
            os.mkdir(nifti_path)
        norm_data = h5py.File(os.path.join('../data/tumor_complete_norm.h5'), 'r')
        if config['norm_type'] == 'z-score':
            data = h5py.File(os.path.join('../data/tumor_complete_zscore.h5'), 'r')
        else:
            data = h5py.File(os.path.join('../data/tumor_complete_mean.h5'), 'r')
        subj_id_list_uni = subj_id_list[0::156]
        # pdb.set_trace()
        for idx, subj_id in enumerate(subj_id_list_uni):
            s_idx = idx * 156
            e_idx = (idx + 1) * 156
            try:
                if slice_idx_list[s_idx] != config['block_size'] and slice_idx_list[e_idx] != 155-config['block_size']:
                    # pdb.set_trace()
                    print(s_idx, e_idx)
                norm = norm_data[subj_id+'/PET']
                if config['norm_type'] == 'z-score':
                    volume = norm[1] * y_fake_fused_list[s_idx: e_idx].squeeze(1) + norm[0]
                else:
                    volume = norm[0] * y_fake_fused_list[s_idx: e_idx].squeeze(1)
                volume = volume[:,:157,:189]
                save_volume_nifti(os.path.join(nifti_path, subj_id + '_pred.nii'), volume)

                if config['norm_type'] == 'z-score':
                    volume = norm[1] * target_list[s_idx: e_idx].squeeze(1) + norm[0]
                else:
                    volume = norm[0] * target_list[s_idx: e_idx].squeeze(1)
                volume = volume[:,:157,:189]
                save_volume_nifti(os.path.join(nifti_path, subj_id + '_PET.nii'), volume)

                for ic, contrast in enumerate(['T1', 'T1c', 'T2_FLAIR', 'ASL']):
                    norm = norm_data[subj_id+'/'+contrast]
                    if config['norm_type'] == 'z-score':
                        volume = norm[1] * np.array(data[subj_id+'/'+contrast][:157,:189]) + norm[0]
                    else:
                        volume = norm[0] * np.array(data[subj_id+'/'+contrast][:157,:189])
                    volume = np.transpose(volume, (2, 0, 1))
                    save_volume_nifti(os.path.join(nifti_path, subj_id + '_'+ contrast + '.nii'), volume)
            except:
                print(subj_id, s_idx, e_idx, len(slice_idx_list))

    return loss_all_dict

if config['phase'] == 'train':
    train()
else:
    stat = evaluate(phase='test', set='test', save_res=True)
    print(stat)
