"""
PointNAT
"""
from re import I
from typing import List, Type
from os.path import exists, join
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..build import MODELS
from ..layers import create_convblock1d, create_convblock2d, create_act, CHANNEL_MAP, \
    create_grouper, furthest_point_sample, random_sample, three_interpolation
import copy, time, math
import matplotlib.pyplot as plt
from einops import rearrange, repeat


num_points = [24000, 6000, 1500, 375, 94]
batch_size = 8


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, num_head=4):
        super().__init__()
        assert out_planes % num_head == 0, "Embedding dimension must be 0 modulo number of heads."
        
        self.mid_planes = out_planes // num_head
        self.out_planes = out_planes
        self.num_head = num_head
        self.linear_q = nn.Linear(in_planes, out_planes)
        self.linear_k = nn.Linear(in_planes, out_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)

        self.linear_values = nn.Linear(out_planes, out_planes)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1, inplace=True)

    def forward(self, f, key_f) -> torch.Tensor:

        b, n, d_in = f.shape
        q = self.linear_q(f)  # (b, n, c')
        k = self.linear_k(key_f)  # (b, n', c')
        v = self.linear_v(key_f)  # (b, n', c')

        x_q = q.reshape(b, -1, self.num_head, self.mid_planes).permute(0, 2, 1, 3)
        x_k = k.reshape(b, -1, self.num_head, self.mid_planes).permute(0, 2, 1, 3)
        x_v = v.reshape(b, -1, self.num_head, self.mid_planes).permute(0, 2, 1, 3)
        
        values, attn = scaled_dot_product(x_q, x_k, x_v)
        values = values.permute(0, 2, 1, 3).reshape(b, n, self.out_planes)
        values = self.dropout(self.linear_values(values))        

        return values.transpose(1, 2)
    
def scaled_dot_product(q, k, v):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)

    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


def get_reduction_fn(reduction):
    reduction = 'mean' if reduction.lower() == 'avg' else reduction
    assert reduction in ['sum', 'max', 'mean']
    if reduction == 'max':
        pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
    elif reduction == 'mean':
        pool = lambda x: torch.mean(x, dim=-1, keepdim=False)
    elif reduction == 'sum':
        pool = lambda x: torch.sum(x, dim=-1, keepdim=False)
    return pool


def get_aggregation_feautres(p, dp, f, fj, feature_type='dp_fj'):
    if feature_type == 'dp_fj':
        fj = torch.cat([dp, fj], 1)
    elif feature_type == 'dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, fj, df], 1)
    elif feature_type == 'pi_dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([p.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, df.shape[-1]), dp, fj, df], 1)
    elif feature_type == 'dp_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, df], 1)
    return fj


class LocalAggregation(nn.Module):
    """Local aggregation layer for a set 
    Set abstraction layer abstracts features from a larger set to a smaller set
    Local aggregation layer aggregates features from the same set
    """

    def __init__(self,
                 channels: List[int],
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 group_args={'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 conv_args=None,
                 feature_type='dp_fj',
                 reduction='max',
                 last_act=True,
                 **kwargs
                 ):
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        channels1 = channels 
        convs1 = []
        for i in range(len(channels1) - 1):  # #layers in each blocks
            convs1.append(create_convblock1d(channels1[i], channels1[i + 1],
                                             norm_args=norm_args,
                                            act_args=None if i == (
                                                    len(channels1) - 2) and not last_act else act_args,
                                             **conv_args)
                          )
        self.convs1 = nn.Sequential(*convs1)
        self.grouper = create_grouper(group_args)
        self.reduction = reduction.lower()
        self.pool = get_reduction_fn(self.reduction)
        self.feature_type = feature_type

    def forward(self, pf, pe) -> torch.Tensor:
        # p: position, f: feature
        p, f = pf
        # preconv
        f = self.convs1(f)
        # grouping
        dp, fj = self.grouper(p, p, f)
        # pe + fj 
        f = pe + fj
        f = self.pool(f)
        """ DEBUG neighbor numbers. 
        if f.shape[-1] != 1:
            query_xyz, support_xyz = p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logging.info(
                f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
        DEBUG end """
        return f


class SetAbstraction(nn.Module):
    """The modified set abstraction module in PointNet++ with residual connection support
    """

    def __init__(self,
                 in_channels, out_channels, i_th,
                 layers=1,
                 stride=1,
                 group_args={'NAME': 'ballquery',
                             'radius': 0.1, 'nsample': 16},
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 conv_args=None,
                 sampler='fps',
                 feature_type='dp_fj',
                 use_res=False,
                 is_head=False,
                 **kwargs, 
                 ):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        self.all_aggr = not is_head and stride == 1
        self.use_res = use_res and not self.all_aggr and not self.is_head
        self.feature_type = feature_type
        self.num_head = 4
        self.i_th = i_th
        group_args.nsample = group_args.nsample//4

        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid_channel] * \
                   (layers - 1) + [out_channels]
        channels[0] = in_channels #if is_head else CHANNEL_MAP[feature_type](channels[0])
        channels1 = channels
        # channels2 = copy.copy(channels)
        channels2 = [in_channels] + [32,32] * (min(layers, 2) - 1) + [out_channels] # 16
        channels2[0] = 3
        convs1 = []
        convs2 = []

        if self.use_res:
            self.skipconv = create_convblock1d(
                in_channels, channels[-1], norm_args=None, act_args=None) if in_channels != channels[
                -1] else nn.Identity()
            self.act = create_act(act_args)

        # actually, one can use local aggregation layer to replace the following
        for i in range(len(channels1) - 1):  # #layers in each blocks
            convs1.append(create_convblock1d(channels1[i], channels1[i + 1],
                                             norm_args=norm_args if not is_head else None,
                                             act_args=None if i == len(channels) - 2
                                                            and (self.use_res or is_head) else act_args,
                                             **conv_args)
                          )
        self.convs1 = nn.Sequential(*convs1)

        grouper = []
        convsp = []
        convsf = []
        convaug = []
        convmix = []
        convmlp = []
        convpool = []
        self.nsample = []
        
        if not is_head:
            # multi-head neighbor aggregation
            self.convpre = create_convblock1d(in_channels, out_channels//(self.num_head),
                                              norm_args=norm_args, act_args=act_args, **conv_args)
            for i in range(self.num_head):
                grouper.append(create_grouper(group_args))

                convsp.append(create_convblock2d(3, out_channels//(self.num_head),
                                                 norm_args=norm_args, act_args=act_args, **conv_args))
                convsf.append(create_convblock2d(out_channels//(self.num_head), out_channels//(self.num_head),
                                                 norm_args=norm_args, act_args=act_args, **conv_args))
                convaug.append(create_convblock2d(out_channels//(2*self.num_head), out_channels//(2*self.num_head),
                                                  norm_args=norm_args, act_args=None, **conv_args))
                convmix.append(create_convblock1d(out_channels//self.num_head, out_channels//self.num_head,
                                                  norm_args=norm_args, act_args=act_args, **conv_args))
                convmlp.append(create_convblock1d(out_channels//self.num_head, out_channels//(self.num_head),
                                                  norm_args=norm_args, act_args=act_args, **conv_args))                
                convpool.append(create_convblock2d(group_args.nsample, 1,
                                                  norm_args=norm_args, act_args=act_args, **conv_args))
                group_args.nsample = group_args.nsample * 2

            self.convsp = nn.Sequential(*convsp)
            self.convsf = nn.Sequential(*convsf)
            self.convaug = nn.Sequential(*convaug)
            self.convmix = nn.Sequential(*convmix)
            self.convmlp = nn.Sequential(*convmlp)
            self.convpool = nn.Sequential(*convpool)
            self.grouper = grouper

            self.pool_1 = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
            self.pool_2 = lambda x: torch.sum(x, dim=-1, keepdim=False)

            self.convfinal = create_convblock1d(out_channels, out_channels, norm_args=norm_args, act_args=act_args, **conv_args)
            self.convmlp_1 = create_mlp(out_channels, out_channels)
            
            # light global point transformer
            self.W = nn.Parameter(torch.randn(num_points[i_th]//10, out_channels))
            self.conv_gp = create_convblock1d(3, out_channels, norm_args=norm_args, act_args=act_args, **conv_args)
            self.conv_g1 = create_convblock1d(out_channels, out_channels, norm_args=norm_args, act_args=act_args, **conv_args)
            group_args.nsample = 1
            self.np_assign = create_grouper(group_args)

            self.pt = PointTransformerLayer(out_channels, out_channels)
            self.np_copy = create_grouper(group_args)
            self.convmlp_2 = create_convblock1d(out_channels, out_channels,
                                                norm_args=norm_args, act_args=act_args, **conv_args)

            self.alpha=nn.Parameter(torch.ones((1,), dtype=torch.float32)*100)
            self.convmlp_3 = create_convblock1d(out_channels, out_channels,
                                                norm_args=norm_args, act_args=act_args, **conv_args)

            if sampler.lower() == 'fps':
                self.sample_fn = furthest_point_sample
            elif sampler.lower() == 'random':
                self.sample_fn = random_sample

    def forward(self, pf_pe):
        p, f, pe = pf_pe
        if self.is_head:
            f = self.convs1(f)  # (n, c)
        else:
            idx = self.sample_fn(p, p.shape[1] // self.stride).long()
            new_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            ###### local ######
            # preconv
            f = self.convpre(f)
            fi = torch.gather(f, -1, idx.unsqueeze(1).expand(-1, f.shape[1], -1))
            Fout = []
            for i in range(self.num_head):
                dp, fj = self.grouper[i](new_p, p, f)
                pe = self.convsp[i](dp)
                fna = pe + self.convsf[i](fj - fi.unsqueeze(-1))
                fmax = self.pool_1(fna)
                fmax = self.convmlp[i](fmax)

                Fout.append(fmax)
            f = torch.concat(Fout, dim=1)
            x = self.convfinal(f)
            f = f + x

            ###### global ######
            local_f = f
            x = self.conv_g1(local_f + self.conv_gp(new_p.transpose(1, 2)))  # b,c,n
            W = repeat(self.W, 'm d -> b m d', b=x.shape[0]).contiguous()
            M = torch.bmm(W, x)  # b,n',c x b,c,n -> b,n',n
            M = M.softmax(-1)
            LP = torch.bmm(M, new_p)  # b,n',n x b,n,3 -> b,n',3
            LP_p, LP_f = self.np_assign(LP, new_p, local_f)
            LP_p, LP_f = LP_p.squeeze(-1).transpose(1,2).contiguous(), LP_f.squeeze(-1)

            LP_gf = self.pt(LP_f.transpose(1,2), LP_f.transpose(1,2))
            _, global_f = self.np_copy(new_p, LP, LP_gf.contiguous())
            # print(global_f.shape)
            x = self.convmlp_2(global_f.squeeze(-1))
            global_f = global_f.squeeze(-1) + x

            ###### Hybrid ######
            alpha = self.alpha.sigmoid()
            f = local_f*(alpha) + global_f*(1-alpha)
            x = self.convmlp_3(f)
            f = f + x

            # if not self.training:
            #     logging.info(alpha)

            p = new_p
        return p, f, pe


class FeaturePropogation(nn.Module):
    """The Feature Propogation module in PointNet++
    """

    def __init__(self, mlp,
                 upsample=True,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'}
                 ):
        """
        Args:
            mlp: [current_channels, next_channels, next_channels]
            out_channels:
            norm_args:
            act_args:
        """
        super().__init__()
        if not upsample:
            self.linear2 = nn.Sequential(
                nn.Linear(mlp[0], mlp[1]), nn.ReLU(inplace=True))
            mlp[1] *= 2
            linear1 = []
            for i in range(1, len(mlp) - 1):
                linear1.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                  norm_args=norm_args, act_args=act_args
                                                  ))
            self.linear1 = nn.Sequential(*linear1)
        else:
            convs = []
            for i in range(len(mlp) - 1):
                convs.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                norm_args=norm_args, act_args=act_args
                                                ))
            self.convs = nn.Sequential(*convs)

        self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

    def forward(self, pf1, pf2=None):
        # pfb1 is with the same size of upsampled points
        if pf2 is None:
            _, f = pf1  # (B, N, 3), (B, C, N)
            f_global = self.pool(f)
            f = torch.cat(
                (f, self.linear2(f_global).unsqueeze(-1).expand(-1, -1, f.shape[-1])), dim=1)
            f = self.linear1(f)
        else:
            p1, f1 = pf1
            p2, f2 = pf2
            if f1 is not None:
                f = self.convs(
                    torch.cat((f1, three_interpolation(p1, p2, f2)), dim=1))
            else:
                f = self.convs(three_interpolation(p1, p2, f2))
        return f


class InvResMLP(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 use_res=True,
                 num_posconvs=2,#2,
                 less_act=False,
                 **kwargs
                 ):
        super().__init__()
        self.use_res = use_res
        mid_channels = int(in_channels * expansion)
        self.convs = LocalAggregation([in_channels, in_channels],
                                      norm_args=norm_args, act_args=act_args ,#if num_posconvs > 0 else None,
                                      group_args=group_args, conv_args=conv_args,
                                      **aggr_args, **kwargs)
        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        elif num_posconvs == 4:
            channels = [in_channels, in_channels, in_channels, in_channels, in_channels]
        elif num_posconvs == 3:
            channels = [in_channels, in_channels, in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]
        pwconv = []
        # point wise after depth wise conv (without last layer)
        for i in range(len(channels) - 1):
            pwconv.append(create_convblock1d(channels[i], channels[i + 1],
                                             norm_args=norm_args,
                                             act_args=act_args if
                                             (i != len(channels) - 2) and not less_act else None,
                                             **conv_args)
                          )
        self.pwconv = nn.Sequential(*pwconv)
        self.act = create_act(act_args)

    def forward(self, pf_pe):
        p, f, pe = pf_pe
        identity = f
        f = self.convs([p, f], pe)
        f = self.pwconv(f)
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f += identity
        f = self.act(f)
        return [p, f, pe]


@MODELS.register_module()
class PointNATEncoder(nn.Module):
    r"""The Encoder for PointNext 
    `"PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies".
    <https://arxiv.org/abs/2206.04670>`_.
    .. note::
        For an example of using :obj:`PointNextEncoder`, see
        `examples/segmentation/main.py <https://github.com/guochengqian/PointNeXt/blob/master/cfgs/s3dis/README.md>`_.
    Args:
        in_channels (int, optional): input channels . Defaults to 4.
        width (int, optional): width of network, the output mlp of the stem MLP. Defaults to 32.
        blocks (List[int], optional): # of blocks per stage (including the SA block). Defaults to [1, 4, 7, 4, 4].
        strides (List[int], optional): the downsampling ratio of each stage. Defaults to [4, 4, 4, 4].
        block (strorType[InvResMLP], optional): the block to use for depth scaling. Defaults to 'InvResMLP'.
        nsample (intorList[int], optional): the number of neighbors to query for each block. Defaults to 32.
        radius (floatorList[float], optional): the initial radius. Defaults to 0.1.
        aggr_args (_type_, optional): the args for local aggregataion. Defaults to {'feature_type': 'dp_fj', "reduction": 'max'}.
        group_args (_type_, optional): the args for grouping. Defaults to {'NAME': 'ballquery'}.
        norm_args (_type_, optional): the args for normalization layer. Defaults to {'norm': 'bn'}.
        act_args (_type_, optional): the args for activation layer. Defaults to {'act': 'relu'}.
        expansion (int, optional): the expansion ratio of the InvResMLP block. Defaults to 4.
        sa_layers (int, optional): the number of MLP layers to use in the SA block. Defaults to 1.
        sa_use_res (bool, optional): wheter to use residual connection in SA block. Set to True only for PointNeXt-S. 
    """

    def __init__(self,
                 in_channels: int = 4,
                 width: int = 32,
                 blocks: List[int] = [1, 4, 7, 4, 4],
                 strides: List[int] = [4, 4, 4, 4],
                 block: str or Type[InvResMLP] = 'InvResMLP',
                 nsample: int or List[int] = 32,
                 radius: float or List[float] = 0.1,
                 aggr_args: dict = {'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args: dict = {'NAME': 'ballquery'},
                 sa_layers: int = 1,
                 sa_use_res: bool = False,
                 **kwargs
                 ):
        super().__init__()
        if isinstance(block, str):
            block = eval(block)
        self.blocks = blocks
        self.strides = strides
        self.in_channels = in_channels
        self.aggr_args = aggr_args
        self.norm_args = kwargs.get('norm_args', {'norm': 'bn'}) 
        self.act_args = kwargs.get('act_args', {'act': 'relu'}) 
        self.conv_args = kwargs.get('conv_args', None)
        self.sampler = kwargs.get('sampler', 'fps')
        self.expansion = kwargs.get('expansion', 4)
        self.sa_layers = sa_layers
        self.sa_use_res = sa_use_res
        self.use_res = kwargs.get('use_res', True)
        radius_scaling = kwargs.get('radius_scaling', 2)
        nsample_scaling = kwargs.get('nsample_scaling', 1)

        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        logging.info(f'radius: {self.radii},\n nsample: {self.nsample}')

        # double width after downsampling.
        channels = []
        for stride in strides:
            if stride != 1:
                width *= 2
            channels.append(width)
        encoder = []
        pe_encoder = nn.ModuleList() #[]
        pe_grouper = []
        group_args.name = 'knn'
        for i in range(len(blocks)):
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            encoder.append(self._make_enc(
                block, channels[i], i, blocks[i], stride=strides[i], group_args=group_args,
                is_head=i == 0 and strides[i] == 1
            ))
            if i == 0:
                pe_encoder.append(nn.ModuleList())
                pe_grouper.append([])
            else:
                pe_encoder.append(self._make_pe_enc(
                    block, channels[i], blocks[i], stride=strides[i], group_args=group_args,
                    is_head=i == 0 and strides[i] == 1
                ))
                pe_grouper.append(create_grouper(group_args))
        self.encoder = nn.Sequential(*encoder)
        self.pe_encoder = pe_encoder #nn.Sequential(*pe_encoder)
        self.pe_grouper = pe_grouper
        self.out_channels = channels[-1]
        self.channel_list = channels

    def _to_full_list(self, param, param_scaling=1):
        # param can be: radius, nsample
        param_list = []
        if isinstance(param, List):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar (in this case, only initial raidus is provide), then create a list (radius for each block)
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def _make_pe_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        ## for PE of this stage
        channels2 = [3, channels]
        convs2 = []
        if blocks > 1:
            for i in range(len(channels2) - 1):  # #layers in each blocks
                convs2.append(create_convblock2d(channels2[i], channels2[i + 1],
                                                norm_args=self.norm_args,
                                                act_args=self.act_args,
                                                **self.conv_args)
                            )
            convs2 = nn.Sequential(*convs2)
            return convs2
        else:
            return nn.ModuleList()

    def _make_enc(self, block, channels, i_th, blocks, stride, group_args, is_head=False):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        group_args.radius = radii[0]
        group_args.nsample = nsample[0]
        layers.append(SetAbstraction(self.in_channels, channels, i_th,
                                     self.sa_layers if not is_head else 1, stride,
                                     group_args=group_args,
                                     sampler=self.sampler,
                                     norm_args=self.norm_args, act_args=self.act_args, conv_args=self.conv_args,
                                     is_head=is_head, use_res=self.sa_use_res, **self.aggr_args 
                                     ))
        self.in_channels = channels
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(block(self.in_channels,
                                aggr_args=self.aggr_args,
                                norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
                                conv_args=self.conv_args, expansion=self.expansion,
                                use_res=self.use_res
                                ))
        return nn.Sequential(*layers)

    def forward_cls_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        for i in range(0, len(self.encoder)):
            pe = None
            p0, f0, pe = self.encoder[i]([p0, f0, pe])
        return f0.squeeze(-1)

    def forward_seg_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        p, f = [p0], [f0]
        for i in range(0, len(self.encoder)):
            if i == 0:
                pe = None
                # pe = f0.transpose(1,-1)
                _p, _f, _ = self.encoder[i]([p[-1], f[-1], pe])
            else:
                _p, _f, _ = self.encoder[i][0]([p[-1], f[-1], pe])
                if self.blocks[i] > 1:
                    # grouping
                    dp, _ = self.pe_grouper[i](_p, _p, None)
                    # conv on neighborhood_dp
                    pe = self.pe_encoder[i](dp)
                    _p, _f, _ = self.encoder[i][1:]([_p, _f, pe])
            p.append(_p)
            f.append(_f)
        return p, f

    def forward(self, p0, f0=None):
        return self.forward_seg_feat(p0, f0)


@MODELS.register_module()
class PointNATDecoder(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 2,
                 decoder_stages: int = 4,
                 **kwargs
                 ):
        super().__init__()
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]
        skip_channels = encoder_channel_list[:-1]
        if len(skip_channels) < decoder_stages:
            skip_channels.insert(0, kwargs.get('in_channels', 3))
        # the output channel after interpolation
        fp_channels = encoder_channel_list[:decoder_stages]

        n_decoder_stages = len(fp_channels)
        decoder = [[] for _ in range(n_decoder_stages)]
        for i in range(-1, -n_decoder_stages - 1, -1):
            decoder[i] = self._make_dec(
                skip_channels[i], fp_channels[i])
        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages]

    def _make_dec(self, skip_channels, fp_channels):
        layers = []
        mlp = [skip_channels + self.in_channels] + \
              [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp))
        self.in_channels = fp_channels
        return nn.Sequential(*layers)

    def forward(self, p, f):
        for i in range(-1, -len(self.decoder) - 1, -1):
            f[i - 1] = self.decoder[i][1:](
                [p[i], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]
        return f[-len(self.decoder) - 1]
