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

from model_simple import *
from util import *

# set seed
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True


_, config = load_config_yaml('config_train.yaml')
config['in_num_ch'] = len(config['contrast_list']) * (2*config['block_size']+1)
config['device'] = torch.device('cuda:'+ config['gpu'])


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


Data = ZeroDoseDataAll(config['dataset_name'], config['data_path'], config['data_h5_path'], config['train_txt_path'], config['val_txt_path'], config['test_txt_path'],
                        batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=0, block_size=config['block_size'], \
                        contrast_list=config['contrast_list'], aug=config['aug'], dropoff=config['dropoff'])
trainDataLoader = Data.trainLoader
valDataLoader = Data.valLoader
# testDataLoader = Data.testLoader


# define model
if config['norm_type'] == 'z-score':
    config['target_output_act'] = 'no'
else:
    config['target_output_act'] = 'softplus'
model = TransUNet(in_num_ch=config['in_num_ch'], out_num_ch=config['out_num_ch'], 
                first_num_ch=64, input_size=(config['input_height'], config['input_width']), 
                is_symmetry=config['is_symmetry'], 
                output_activation=config['target_output_act'], is_transformer=config['is_transformer']).to(config['device'])

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
scheduler = PolyLR(optimizer, max_epoch=50, power=0.9)

if config['is_gan']:
    discriminator = Discriminator(in_num_ch=config['in_num_ch']+config['out_num_ch'], inter_num_ch=64, input_shape=(config['input_height'], config['input_width']), is_patch_gan=True).to(config['device'])
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config['lr'], betas=(0.5, 0.999))

# load pretrained model
if config['continue_train']:
    [optimizer, scheduler, model], start_epoch = load_checkpoint_by_key([optimizer, scheduler, model], config['ckpt_path'], ['optimizer', 'scheduler', 'model'], config['device'], config['ckpt_name'])
    # [optimizer, model], start_epoch = load_checkpoint_by_key([optimizer, model], config['ckpt_path'], ['optimizer', 'model'], config['device'], config['ckpt_name'])
    # [model], start_epoch = load_checkpoint_by_key([model], config['ckpt_path'], ['model'], config['device'], config['ckpt_name'])
else:
    start_epoch = -1

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

            if config['is_gan']:
                output_d_real = discriminator(torch.cat([inputs, targets], dim=1))
                output_d_fake = discriminator(torch.cat([inputs, pred.detach()], dim=1))
                loss_d = 0.5 * (F.binary_cross_entropy_with_logits(output_d_real, torch.ones_like(output_d_real)) + \
                        F.binary_cross_entropy_with_logits(output_d_fake, torch.zeros_like(output_d_fake)))
                optimizer_D.zero_grad()
                loss_d.backward()
                optimizer_D.step()
                

            if config['p'] == 1:
                loss_recon_y = torch.abs(pred - targets).mean()
            else:
                loss_recon_y = torch.pow(pred - targets, 2).mean()
            loss = config['lambda_recon_y'] * loss_recon_y

            if config['lambda_gan'] > 0 and config['is_gan'] == 'GAN':
                output_d_fake = discriminator(torch.cat([inputs, pred], dim=1))
                loss_gan = F.binary_cross_entropy_with_logits(output_d_fake.detach(), torch.ones_like(output_d_fake))
                loss += config['lambda_gan'] * loss_gan
                print(loss_recon_y, loss_gan, loss_d)

            loss_all_dict['recon_y'] += loss_recon_y.item()
            loss_all_dict['all'] += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for name, param in model.named_parameters():
                try:
                    if not torch.isfinite(param.grad).all():
                        pdb.set_trace()
                except:
                    continue

            optimizer.step()
            

            if global_iter % 10 == 0:
                print('Epoch[%3d], iter[%3d]: loss=[%.4f], recon y=[%.4f]' \
                        % (epoch, iter, loss.item(), loss_recon_y.item()))

            # if iter > 5:
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
        else:
            raise ValueError('Undefined loader')

    loss_all_dict = {'recon_y': 0., 'all': 0.}

    subj_id_list = []
    slice_idx_list = []
    input_list = []
    target_list = []
    mask_list = []
    y_fake_fused_list = []
    metrics_list_dict = {}

    res_path = os.path.join(config['ckpt_path'], 'result_'+set+info)
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

            #
            # if iter > 5:
            #     break

    for key in loss_all_dict.keys():
        loss_all_dict[key] /= (iter + 1)

    for key in metrics_list_dict:
        loss_all_dict[key] = np.array(metrics_list_dict[key]).mean()

    return loss_all_dict

train()

