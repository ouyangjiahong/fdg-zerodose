import pdb

import torch
# import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.nn.modules.utils import _pair
from torch.nn.modules.conv import _ConvNd
import functools
import numpy as np
# from util import *


'''
Transformer
'''
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_head=4, num_feat_in=32, num_feat_emb=16, num_feat_out=32):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_head = num_head
        self.num_feat_in = num_feat_in
        self.num_feat_emb = num_feat_emb
        self.num_feat_fused = num_head * num_feat_emb
        self.num_feat_out = num_feat_out

        self.norm = nn.LayerNorm([num_feat_in])
        self.W_q = nn.Linear(num_feat_in, self.num_feat_fused)
        self.W_k = nn.Linear(num_feat_in, self.num_feat_fused)
        self.W_v = nn.Linear(num_feat_in, self.num_feat_fused)
        self.W_o = nn.Linear(self.num_feat_fused, self.num_feat_out)

    def compute_attention(self, Q, K, V):
        qk = torch.matmul(Q.transpose(2,1), K.transpose(2,1).transpose(-1,-2)) # (bs, num_head, ts, ts)
        attn = torch.softmax(qk / torch.sqrt(torch.tensor(self.num_feat_emb).float()), dim=-1) # (bs, num_head, ts, ts)
        out = torch.matmul(attn, V.transpose(2,1))  # (bs, num_head, ts, num_feat_emb)
        return out

    def forward(self, X):
        bs, ts, fs = X.shape
        X = self.norm(X)
        Q = self.W_q(X).view(bs, ts, self.num_head, -1)
        K = self.W_k(X).view(bs, ts, self.num_head, -1)
        V = self.W_v(X).view(bs, ts, self.num_head, -1)   # (bs, ts, num_head, num_feat_emb)
        X_attn = self.compute_attention(Q, K, V).transpose(2,1).contiguous().view(bs, ts, -1) # (bs, ts, num_head*num_feat_emb)
        return self.W_o(X_attn)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, num_block=4, num_head=4, num_feat_in=32, num_feat_emb=16, num_feat_mlp=32, dropout=0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_block):
            self.layers.append(nn.ModuleList([
                MultiHeadSelfAttention(num_head, num_feat_in, num_feat_emb, num_feat_in),
                FeedForward(num_feat_in, num_feat_mlp, dropout=dropout)
            ]))
        self.norm = nn.LayerNorm([num_feat_in])
        
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

'''
Conv block
'''
class Conv_BN_Act(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, filter_size=4, stride=2, padding=1, activation='lrelu', is_bn=True):
        super(Conv_BN_Act, self).__init__()
        if is_bn:
            self.conv = nn.Sequential(
                nn.Conv2d(in_num_ch, out_num_ch, filter_size, stride, padding=padding),
                nn.BatchNorm2d(out_num_ch)
                )
        else:
            self.conv = nn.Conv2d(in_num_ch, out_num_ch, filter_size, stride, padding=padding)
        if activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        if activation == 'elu':
            self.act = nn.ELU(inplace=True)
        else:
            self.act = nn.Sequential()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x

class Act_Deconv_BN_Concat(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, filter_size=3, stride=1, padding=1, activation='relu', upsample=True, is_last=False, is_bn=True):
        super(Act_Deconv_BN_Concat, self).__init__()
        self.is_bn = is_bn
        if activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        if activation == 'elu':
            self.act = nn.ELU(inplace=True)
        else:
            self.act = nn.Sequential()
        self.is_last = is_last

        if upsample == True:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_num_ch, out_num_ch, filter_size, stride, padding=padding)
                )
        else:
            self.up = nn.ConvTranspose2d(in_num_ch, out_num_ch, filter_size, padding=padding, stride=stride) # (H-1)*stride-2*padding+kernel_size
        self.bn = nn.BatchNorm2d(out_num_ch)

    def forward(self, x_down, x_up):
        # pdb.set_trace()
        x_up = self.act(x_up)
        x_up = self.up(x_up)
        if self.is_last == False:
            if self.is_bn:
                x_up = self.bn(x_up)
            x = torch.cat([x_down, x_up], 1)
        else:
            x = x_up
        return x
    
'''
Attention block
'''
class ChannelAttentionLayer(nn.Module):
    # CVPR2018 squeeze and excitation
    def __init__(self, in_num_ch, sample_factor=16):
        super(ChannelAttentionLayer, self).__init__()

        self.W_down = nn.Linear(in_num_ch, in_num_ch//sample_factor)
        self.W_up = nn.Linear(in_num_ch//sample_factor, in_num_ch)

    def forward(self, x):
        x_gp = torch.mean(x, (2,3))

        x_down = F.relu(self.W_down(x_gp))
        alpha = F.sigmoid(self.W_up(x_down))

        alpha_exp = alpha.unsqueeze(2).unsqueeze(3).expand_as(x)
        out = (1 + alpha_exp) * x
        return out, alpha
    
class SpatialAttentionLayer(nn.Module):
    def __init__(self, in_num_ch, gate_num_ch, inter_num_ch, sample_factor=(2,2)):
        super(SpatialAttentionLayer, self).__init__()

        # in_num_ch, out_num_ch, kernel_size, stride, padding
        self.W_x = nn.Conv2d(in_num_ch, inter_num_ch, sample_factor, sample_factor, bias=False)
        self.W_g = nn.Conv2d(gate_num_ch, inter_num_ch, 1, 1)
        self.W_psi = nn.Conv2d(inter_num_ch, 1, 1, 1)
        self.W_out = nn.Sequential(
            nn.Conv2d(in_num_ch, in_num_ch, 1, 1),
            nn.BatchNorm2d(in_num_ch)
        )

    def forward(self, x, g):
        x_size = x.size()
        x_post = self.W_x(x)
        x_post_size = x_post.size()

        g_post = F.upsample(self.W_g(g), size=x_post_size[2:], mode='bilinear')
        xg_post = F.relu(x_post + g_post, inplace=True)
        alpha = F.sigmoid(self.W_psi(xg_post))
        alpha_upsample = F.upsample(alpha, size=x_size[2:], mode='bilinear')

        out = self.W_out(alpha_upsample * x)
        return out, alpha_upsample
    
class SymmetryGateResidualSpatialAttentionLayer(nn.Module):
    # only g
    def __init__(self, in_num_ch, gate_num_ch, inter_num_ch, sample_factor=(2,2), is_bn=True):
        super(SymmetryGateResidualSpatialAttentionLayer, self).__init__()

        # in_num_ch, out_num_ch, kernel_size, stride, padding
        self.W_g = nn.Conv2d(gate_num_ch, inter_num_ch, 1, 1)
        self.W_g_diff = nn.Conv2d(gate_num_ch, inter_num_ch, 1, 1)
        self.W_psi = nn.Conv2d(inter_num_ch, 1, 1, 1)
        if is_bn:
            self.W_out = nn.Sequential(
                nn.Conv2d(in_num_ch, in_num_ch, 1, 1),
                nn.BatchNorm2d(in_num_ch)
            )
        else:
            self.W_out = nn.Conv2d(in_num_ch, in_num_ch, 1, 1)

    def forward(self, x, g):
        x_size = x.size()
        g_flip = torch.flip(g, dims=[2])
        g_diff = torch.abs(g - g_flip)
        g_post = F.relu(self.W_g(g) + self.W_g_diff(g_diff), inplace=True)
        alpha = F.sigmoid(self.W_psi(g_post))
        alpha_upsample = F.upsample(alpha, size=x_size[2:], mode='bilinear')

        out = self.W_out((1+alpha_upsample) * x)
        return out, alpha_upsample

'''
TransUNet with Attention
'''
class TransUNet(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, first_num_ch=64, input_size=(256,256), sample_factor=(2,2), is_symmetry=True, output_activation='softplus', is_transformer=False):
        super(TransUNet, self).__init__()

        self.down_1 = nn.Sequential(
                nn.Conv2d(in_num_ch, first_num_ch, 4, 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.down_2 = Conv_BN_Act(first_num_ch, 2*first_num_ch)
        self.down_3 = Conv_BN_Act(2*first_num_ch, 4*first_num_ch)
        self.down_4 = Conv_BN_Act(4*first_num_ch, 8*first_num_ch)
        self.down_5 = Conv_BN_Act(8*first_num_ch, 8*first_num_ch, activation='no') # 8 x 8

        if is_symmetry:
            spatial_layer = SymmetryGateResidualSpatialAttentionLayer
        else:
            spatial_layer = SpatialAttentionLayer

        self.att_4_c = ChannelAttentionLayer(8*first_num_ch, 8)
        self.att_4_s = spatial_layer(8*first_num_ch, 8*first_num_ch, 8*first_num_ch, sample_factor)
        self.up_4 = Act_Deconv_BN_Concat(8*first_num_ch, 8*first_num_ch)

        self.att_3_c = ChannelAttentionLayer(4*first_num_ch, 4)
        self.att_3_s = spatial_layer(4*first_num_ch, 16*first_num_ch, 4*first_num_ch, sample_factor)
        self.up_3 = Act_Deconv_BN_Concat(16*first_num_ch, 4*first_num_ch)

        self.att_2_c = ChannelAttentionLayer(2*first_num_ch, 2)
        self.att_2_s = spatial_layer(2*first_num_ch, 8*first_num_ch, 2*first_num_ch, sample_factor)
        self.up_2 = Act_Deconv_BN_Concat(8*first_num_ch, 2*first_num_ch)

        self.att_1_c = ChannelAttentionLayer(first_num_ch, 1)
        self.att_1_s = spatial_layer(first_num_ch, 4*first_num_ch, first_num_ch, sample_factor)
        self.up_1 = Act_Deconv_BN_Concat(4*first_num_ch, first_num_ch)
        self.output = Act_Deconv_BN_Concat(2*first_num_ch, out_num_ch, is_last=True)

        # choose different activation layer
        if output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_act = nn.Tanh()
        elif output_activation == 'no':
            self.output_act = nn.Sequential()
        else:
            self.output_act = nn.Softplus()

        self.is_transformer = is_transformer
        if self.is_transformer:
            # num_patches = input_size[0]//32 * input_size[1]//32
            self.pos_embedding = nn.Parameter(torch.randn(1, 8*first_num_ch, input_size[0]//32, input_size[1]//32))
            self.transformer = Transformer(num_block=4, num_head=4, num_feat_in=8*first_num_ch, num_feat_emb=first_num_ch, num_feat_mlp=4*first_num_ch, dropout=0.2)
            # self.transformer = Transformer(num_block=12, num_head=4, num_feat_in=8*first_num_ch, num_feat_emb=first_num_ch, num_feat_mlp=4*first_num_ch, dropout=0.2)


    def forward(self, x):
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)

        if self.is_transformer:
            down_5 += self.pos_embedding
            bs, fs, h, w = down_5.shape
            down_5 = down_5.view(bs, fs,-1).transpose(1,2)
            down_5 = self.transformer(down_5)
            down_5 = down_5.transpose(1,2).view(bs, fs, h, w)

        concat_4_c, _ = self.att_4_c(down_4)
        concat_4_s, alpha_4 = self.att_4_s(down_4, down_5)
        up_4 = self.up_4(concat_4_c+concat_4_s, down_5)

        concat_3_c, _ = self.att_3_c(down_3)
        concat_3_s, alpha_3 = self.att_3_s(down_3, up_4)
        up_3 = self.up_3(concat_3_c+concat_3_s, up_4)

        concat_2_c, _ = self.att_2_c(down_2)
        concat_2_s, alpha_2 = self.att_2_s(down_2, up_3)
        up_2 = self.up_2(concat_2_c+concat_2_s, up_3)

        concat_1_c, _ = self.att_1_c(down_1)
        concat_1_s, alpha_1 = self.att_1_s(down_1, up_2)
        up_1 = self.up_1(concat_1_c+concat_1_s, up_2)

        output = self.output(None, up_1)
        output_act = self.output_act(output)
        return output_act, {'alpha_4':alpha_4, 'alpha_3':alpha_3, 'alpha_2':alpha_2, 'alpha_1':alpha_1}


class Discriminator(nn.Module):
    def __init__(self, in_num_ch=8, inter_num_ch=16, input_shape=[160,192], is_patch_gan=False):
        super(Discriminator, self).__init__()
        self.discrim = nn.Sequential(
                            nn.Conv2d(in_num_ch, inter_num_ch, 4, 2, padding=1),
                            nn.LeakyReLU(0.2),
                            nn.Conv2d(inter_num_ch, 2*inter_num_ch, 4, 2, padding=1),
                            nn.BatchNorm2d(2*inter_num_ch),
                            nn.LeakyReLU(0.2),
                            nn.Conv2d(2*inter_num_ch, 4*inter_num_ch, 4, 2, padding=1),
                            nn.BatchNorm2d(4*inter_num_ch),
                            nn.LeakyReLU(0.2),
                            nn.Conv2d(4*inter_num_ch, 8*inter_num_ch, 4, 2, padding=1),
                            nn.BatchNorm2d(8*inter_num_ch),
                            nn.LeakyReLU(0.2),
                            nn.Conv2d(8*inter_num_ch, 4*inter_num_ch, 4, 2, padding=1),
                            nn.BatchNorm2d(4*inter_num_ch),
                            nn.LeakyReLU(0.2))
                            # nn.Tanh())
        if is_patch_gan:
            self.fc = nn.Conv2d(4*inter_num_ch, 1, 3, 1, padding=1)
        else:
            self.fc = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(int(input_shape[0]*input_shape[1]*4*inter_num_ch/(32*32)), inter_num_ch*16),
                                nn.LeakyReLU(0.2),
                                nn.Linear(inter_num_ch*16, 1))

    def forward(self, x):
        conv = self.discrim(x)
        fc = self.fc(conv)
        return fc
