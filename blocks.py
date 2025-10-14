import numbers

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
import pywt
import pywt.data
from functools import partial
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply

#_______________________________________________
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)



class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)#1,65536,32
        q_inp = self.to_q(x)#,1,66536,dim_head * heads
        k_inp = self.to_k(x)#,1,66536,dim_head * heads
        v_inp = self.to_v(x)#,1,66536,dim_head * heads
        illu_attn = illu_fea_trans # illu_fea: b,c,h,w -> b,h,w,c#1,256,256,32
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n d -> b 1 n d'),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))#b,num_heads,65536,头围
        #要让illu_attn的c等于
        v = v * illu_attn
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.dim)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    def __init__(
            self,
            dim,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out



class Illumination_Estimator(nn.Module):
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):  # __init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img,mean_c):  #输入的是反射率图 和 照度图
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        # stx()
        input = torch.cat([img, mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map

#_________________________________________________________________
class ExpandedDB(nn.Module):
    def __init__(self, in_channel=64, inter_num=32):
        super().__init__()
        # 原始d_list=[1,2,1]对应的三个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, inter_num, 3, padding=1, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel + inter_num, inter_num, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel + 2 * inter_num, inter_num, 3, padding=1, dilation=1),
            nn.ReLU(inplace=True)
        )

        # 最终融合层（显式计算总通道数）
        total_channels = in_channel + 3 * inter_num  # 64 + 3*32 = 160
        self.final_conv = nn.Conv2d(total_channels, in_channel, 1)

    def forward(self, x):
        # 第一层处理
        c1 = self.conv1(x)  # [B,32,H,W]
        cat1 = torch.cat([x, c1], 1)  # [B,96,H,W]

        # 第二层处理
        c2 = self.conv2(cat1)  # [B,32,H,W]
        cat2 = torch.cat([cat1, c2], 1)  # [B,128,H,W]

        # 第三层处理
        c3 = self.conv3(cat2)  # [B,32,H,W]
        cat3 = torch.cat([cat2, c3], 1)  # [B,160,H,W]

        # 最终融合
        out = self.final_conv(cat3)
        return out + x  # 保持残差连接

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class SAM(nn.Module):
    def __init__(self, in_channel=64):
        super().__init__()
        # 三个尺度的DB块
        self.db0 = ExpandedDB(in_channel)
        self.db2 = ExpandedDB(in_channel)
        self.db4 = ExpandedDB(in_channel)

        # 注意力融合模块
        self.csaf = CSAF(3 * in_channel)

    def forward(self, x):
        # 原尺度分支
        y0 = self.db0(x)

        # 1/2尺度分支
        x2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        y2 = self.db2(x2)
        y2 = F.interpolate(y2, size=x.shape[2:], mode='bilinear')

        # 1/4尺度分支
        x4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        y4 = self.db4(x4)
        y4 = F.interpolate(y4, size=x.shape[2:], mode='bilinear')

        # 注意力融合
        fused = self.csaf(y0, y2, y4)
        return x + fused  # 残差连接

class CSAF(nn.Module):
    def __init__(self, in_chnls, ratio=4):
        super(CSAF, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress1 = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.compress2 = nn.Conv2d(in_chnls // ratio, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x0, x2, x4):
        out0 = self.squeeze(x0)
        out2 = self.squeeze(x2)
        out4 = self.squeeze(x4)
        out = torch.cat([out0, out2, out4], dim=1)
        out = self.compress1(out)
        out = F.relu(out)
        out = self.compress2(out)
        out = F.relu(out)
        out = self.excitation(out)
        out = F.sigmoid(out)
        w0, w2, w4 = torch.chunk(out, 3, dim=1)
        x = x0 * w0 + x2 * w2 + x4 * w4

        return x
#________________________________________________________

#________________________________________________________

class FFCSE_block(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * \
            self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * \
            self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()

        # (batch, c, h, w/2+1, 2)
        # ffted = torch.rfft(x, signal_ndim=2, normalized=True)
        # 修改为
        ffted = torch.fft.rfft2(x, norm='ortho')  # 2D实数FFT
        ffted = torch.view_as_real(ffted)  # [B, C, H, W//2+1, 2]
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)

        # output = torch.irfft(ffted, signal_ndim=2,signal_sizes=r_size[2:], normalized=True)
        # 新版PyTorch修改方案
        output = torch.fft.irfft2(
            torch.view_as_complex(ffted),  # 确保输入是复数张量
            s=r_size[2:],  # 原始图像尺寸
            norm='ortho'  # 对应旧版normalized=True
        )
        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 ratio_gin=0.5, ratio_gout=0.5, stride=1, padding=1,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)#两个都是bchw的张量
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3,
                 ratio_gin=0.5,  # 输入通道中全局特征（global）的占比，范围[0,1]
                 ratio_gout=0.5,  # 输出通道中全局特征（global）的占比，范围[0,1]
                 stride=1,  # 卷积步长（默认为1，保持分辨率）
                 padding=1,  # 输入填充量（需根据kernel_size调整）
                 dilation=1,  # 空洞卷积率（默认为1，标准卷积）
                 groups=1,  # 分组卷积数（通常与通道数配合使用）
                 bias=False,  # 是否使用偏置项
                 norm_layer=nn.BatchNorm2d,  # 归一化层类型（可替换为InstanceNorm等）
                 activation_layer=nn.Identity,  # 激活函数类型（如nn.ReLU, nn.GELU）
                 enable_lfu=True  # 是否启用低频成分处理（通常保持True）
                 ):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_l = lnorm(int(out_channels * (1 - ratio_gout)))
        self.bn_g = gnorm(int(out_channels * ratio_gout))

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFC_ResBlock(nn.Module):
    """基于FFC的残差块，支持频域与空域混合处理"""

    def __init__(self, channels, ratio_g=0.5, kernel_size=3, expansion=4):
        """
        channels: 输入输出总通道数
        ratio_g: 全局通道占比(0-1)
        kernel_size: 空间卷积核尺寸
        expansion: 中间层通道扩展倍数
        """
        super().__init__()
        self.ratio_g = ratio_g
        hidden_channels = channels * expansion

        # 通道分割参数
        self.in_cl = int(channels * (1 - ratio_g))  # 输入局部通道
        self.in_cg = channels - self.in_cl  # 输入全局通道

        # 第一个FFC处理单元
        self.ffc1 = FFC_BN_ACT(
            in_channels=channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            ratio_gin=ratio_g,
            ratio_gout=ratio_g,
            activation_layer=nn.ReLU
        )

        # 频域通道注意力
        self.se = FFCSE_block(hidden_channels, ratio_g)

        # 第二个FFC处理单元
        self.ffc2 = FFC_BN_ACT(
            in_channels=hidden_channels,
            out_channels=channels,
            kernel_size=kernel_size,
            ratio_gin=ratio_g,
            ratio_gout=ratio_g,
            activation_layer=nn.Identity  # 最后激活在残差连接后
        )

        # 自适应通道对齐（当expansion≠1时可能需要）
        self.shortcut = nn.Identity()

        # 最终激活函数
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # 输入格式统一处理
        if not isinstance(x, tuple):
            x_l = x[:, :self.in_cl, ...]  # 局部特征切片
            x_g = x[:, self.in_cl:, ...]  # 全局特征切片
            x = (x_l, x_g)

        identity = (x[0], x[1])#x[0]是局部特征，x[1]是全局特征

        # 主体处理流程
        out = self.ffc1(x)  # [B, hidden, ...]
        out = self.se(out)  # 频域注意力增强
        out = self.ffc2(out)  # [B, channels, ...]

        # 残差连接
        out_l = out[0] + self.shortcut(identity[0])
        out_g = out[1] + self.shortcut(identity[1])

        # 激活与合并
        out_l = self.act(out_l)
        out_g = self.act(out_g)

        return (out_l, out_g)


class FFC_ResNet_Converter(nn.Module):
    """将FFC残差块转换为标准张量输出"""

    def __init__(self, in_channels, ratio_g=0.5):
        super().__init__()
        self.ratio_g = ratio_g
        self.out_cl = int(in_channels * (1 - ratio_g))
        self.out_cg = in_channels - self.out_cl

    def forward(self, x):
        if isinstance(x, tuple):
            # 合并局部和全局特征
            return torch.cat([x[0], x[1]], dim=1)
        return x
#_________________________________________________
class GaussianBlur(nn.Module):
    """可微分高斯模糊层，用于强制光照平滑性"""

    def __init__(self, kernel_size=15, sigma=3):
        super().__init__()
        self.kernel = self.create_gaussian_kernel(kernel_size, sigma)

    def create_gaussian_kernel(self, size, sigma):
        """创建高斯卷积核"""
        coords = torch.arange(size).float() - (size - 1) / 2.0
        x = coords.reshape(-1, 1).repeat(1, size)
        y = coords.reshape(1, -1).repeat(size, 1)
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)  # [1,1,H,W]

    def forward(self, x):
        # x: [B, C, H, W]
        channels = x.size(1)
        # 对每个通道应用高斯模糊
        blurred = []
        for c in range(channels):
            blurred_ch = F.conv2d(x[:, c:c + 1],
                                  self.kernel.to(x.device),
                                  padding=(self.kernel.size(2) // 2,
                                           self.kernel.size(3) // 2))
            blurred.append(blurred_ch)
        return torch.cat(blurred, dim=1)

#_____________________________________________________________________

##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## TDB模块
class FeedForward_1(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x
#
#
# ##########################################################################
# ## MDLA improved by designing Multi-scale Convolution
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))


        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        self.qkv_dwconv_3 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.qkv_dwconv_5 = nn.Conv2d(dim * 3, dim * 3, kernel_size=5, stride=1, padding=2, groups=dim * 3, bias=bias)
        self.qkv_dwconv_7 = nn.Conv2d(dim * 3, dim * 3, kernel_size=7, stride=1, padding=3, groups=dim * 3, bias=bias)

        self.q_proj = nn.Conv2d(dim * 3, dim, kernel_size=1 ,stride=1, padding=0, bias=bias)
        self.k_proj = nn.Conv2d(dim * 3, dim, kernel_size=1, stride=1, padding=0, bias=bias)
        self.v_proj = nn.Conv2d(dim * 3, dim, kernel_size=1, stride=1, padding=0, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)



    def forward(self, x):
        b, c, h, w = x.shape

        x = self.qkv(x)
        qkv_1 = self.qkv_dwconv_3(x)
        q_1, k_1, v_1 = qkv_1.chunk(3, dim=1)

        qkv_2 = self.qkv_dwconv_5(x)
        q_2, k_2, v_2 = qkv_2.chunk(3, dim=1)

        qkv_3 = self.qkv_dwconv_7(x)
        q_3, k_3, v_3 = qkv_3.chunk(3, dim=1)

        q = self.q_proj(torch.cat([q_1, q_2, q_3], dim=1))
        k = self.k_proj(torch.cat([k_1, k_2, k_3], dim=1))
        v = self.v_proj(torch.cat([v_1, v_2, v_3], dim=1))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)


        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)


        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads = 4, ffn_expansion_factor =2, bias = True, LayerNorm_type = 'BiasFree'):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward_1(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class EfficientTransformerBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 下采样层
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),  # 下采样1/2
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 低分辨率处理
        self.trans = nn.Sequential(
            TransformerBlock(64),
            TransformerBlock(64)
        )

        # 上采样恢复
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, in_channels, 3, padding=1)
        )

        self.final = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        identity = x  # 保存输入特征

        # 下采样路径
        x = self.downsample(x)  # [B,64,H/2,W/2]
        x = self.trans(x)  # 在低分辨率处理
        x = self.upsample(x)  # [B,64,H,W]

        # 残差连接
        return self.final(x + identity)  # 融合高低频信息
#———————————————————————————————————————————————————TDB模块



#CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化和最大池化（在通道维度上）
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(combined)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # 通道注意力
        x = x * self.channel_attention(x)
        # 空间注意力
        x = x * self.spatial_attention(x)
        return x

#小波卷积
def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters#512,1,2,2


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)

    return x
# def wavelet_transform(x, filters):
#     b, c, h, w = x.shape
#     pad = (filters.shape[2]//2 - 1, filters.shape[3]//2 - 1)
#     x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
#     # 获取卷积后的实际尺寸
#     _, _, new_h, new_w = x.shape
#     # 使用新的尺寸进行reshape
#     x = x.reshape(b, c, 4, new_h, new_w)
#     return x

def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)#创建一个只需x的新函数
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

#mspablock，用多尺度替换残差块的3*3卷积
class SPRModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SPRModule, self).__init__()

        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)

        self.fc1 = nn.Conv2d(channels * 5, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out1 = self.avg_pool1(x).view(x.size(0), -1, 1, 1)
        out2 = self.avg_pool2(x).view(x.size(0), -1, 1, 1)
        out = torch.cat((out1, out2), 1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def convdilated(in_planes, out_planes, kSize=3, stride=1, dilation=1):
    """3x3 convolution with dilation"""
    padding = int((kSize - 1) / 2) * dilation
    return nn.Conv2d(in_planes, out_planes, kernel_size=kSize, stride=stride, padding=padding,
                     dilation=dilation, bias=False)


class MSAModule(nn.Module):
    def __init__(self, inplanes, scale=3, stride=1, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality.
            scale: number of scale.
            stride: conv stride.
            stype: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(MSAModule, self).__init__()

        self.width = inplanes
        self.nums = scale
        self.stride = stride
        assert stype in ['stage', 'normal'], 'One of these is suppported (stage or normal)'
        self.stype = stype

        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])

        for i in range(self.nums):
            if self.stype == 'stage' and self.stride != 1:
                self.convs.append(convdilated(self.width, self.width, stride=stride, dilation=int(i + 1)))
            else:
                self.convs.append(conv3x3(self.width, self.width, stride))

            self.bns.append(nn.BatchNorm2d(self.width))

        self.attention = SPRModule(self.width)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]

        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0 or (self.stype == 'stage' and self.stride != 1):
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.bns[i](sp)

            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        feats = out
        feats = feats.view(batch_size, self.nums, self.width, feats.shape[2], feats.shape[3])

        sp_inp = torch.split(out, self.width, 1)

        attn_weight = []
        for inp in sp_inp:
            attn_weight.append(self.attention(inp))

        attn_weight = torch.cat(attn_weight, dim=1)
        attn_vectors = attn_weight.view(batch_size, self.nums, self.width, 1, 1)
        attn_vectors = self.softmax(attn_vectors)
        feats_weight = feats * attn_vectors

        for i in range(self.nums):
            x_attn_weight = feats_weight[:, i, :, :, :]
            if i == 0:
                out = x_attn_weight
            else:
                out = torch.cat((out, x_attn_weight), 1)

        return out


class MSPABlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=30, scale=3,
                 norm_layer=None, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality.
            planes: output channel dimensionality.
            stride: conv stride.
            downsample: None when stride = 1.
            baseWidth: basic width of conv3x3.
            scale: number of scale.
            norm_layer: regularization layer.
            stype: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(MSPABlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(math.floor(planes * (baseWidth / 64.0)))

        self.conv1 = conv1x1(inplanes, width * scale)
        self.bn1 = norm_layer(width * scale)

        self.conv2 = MSAModule(width, scale=scale, stride=stride, stype=stype)
        self.bn2 = norm_layer(width * scale)

        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

#空间注意力多尺度高效解码器
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


#   Multi-scale depth-wise convolution (MSDC)
class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        # You can return outputs based on what you intend to do with them
        return outputs


class MSCB(nn.Module):
    """
    Multi-scale convolution block (MSCB)
    """

    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
                 add=True, activation='relu6'):
        super(MSCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation,
                         dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels * 1
        else:
            self.combined_channels = self.ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


#   Multi-scale convolution block (MSCB)
def MSCBLayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
              add=True, activation='relu6'):
    """
    create a series of multi-scale convolution blocks.
    """
    convs = []
    mscb = MSCB(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                dw_parallel=dw_parallel, add=add, activation=activation)
    convs.append(mscb)
    if n > 1:
        for i in range(1, n):
            mscb = MSCB(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                        dw_parallel=dw_parallel, add=add, activation=activation)
            convs.append(mscb)
    conv = nn.Sequential(*convs)
    return conv


#   Channel attention block (CAB)
class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)

    #   Spatial attention block (SAB)


class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class MSCAM(nn.Module):
    def __init__(self,channels=64, kernel_sizes=[1,3,5], expansion_factor=6, dw_parallel=True, add=True, activation='relu6'):
        super().__init__()
        self.cab = CAB(channels)
        self.sab = SAB()

        self.mscb = MSCBLayer(channels, channels, n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)
    def forward(self, x):
        d = x
        # MSCAM
        d = self.cab(d) * d
        d = self.sab(d) *d
        d = self.mscb(d)
        return  d


#ela高效局部注意力机制，聚焦空间重点区域
class EfficientLocalizationAttention(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(EfficientLocalizationAttention, self).__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=self.pad, groups=channel, bias=False)
        self.gn = nn.GroupNorm(16, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # 处理高度维度
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)

        # 处理宽度维度
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)

        # 在两个维度上应用注意力
        return x * x_h * x_w


#多尺度线性注意力机制
class MSELA(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(MSELA, self).__init__()
        # 四层下采样对应的注意力模块
        self.ela_layers = nn.ModuleList([
            EfficientLocalizationAttention(channel, kernel_size)
            for _ in range(4)  # 对应原图、2x、4x、8x下采样
        ])

        # 可学习的特征权重
        self.weights = nn.Parameter(torch.ones(4))

    def forward(self, x):
        # 生成四个层级的特征图
        pyramid = [x]  # 原图
        for _ in range(3):  # 递归下采样三次
            pyramid.append(F.max_pool2d(pyramid[-1], kernel_size=2, stride=2))

        # 各层级独立处理
        processed = []
        for i, feat in enumerate(pyramid):
            # 注意力处理
            attn_feat = self.ela_layers[i](feat)

            # 对齐到原图尺寸（当i>0时需要上采样）
            if i > 0:
                attn_feat = F.interpolate(
                    attn_feat,
                    size=x.shape[2:],
                    mode='bilinear',
                    align_corners=True
                )
            processed.append(attn_feat)

        # 加权融合（使用softmax归一化权重）
        weights = F.softmax(self.weights, dim=0)
        x_final = sum([w * feat for w, feat in zip(weights, processed)])

        return x_final

#AFF注意力特征融合模块
class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        # 局部注意力(压缩通道再恢复通道实现的）
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        xa = x + y
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = x * wei + y * (1 - wei)
        return xo
#金字塔注意力特征融合模块
class MSAFB(nn.Module):

    def __init__(self, channels=64, r=4):
        super().__init__()
        inter_channels = int(channels // r)

        # 局部注意力(压缩通道再恢复通道实现的）
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.ppf  = PyramidPoolingFusion(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x1 = torch.cat([x, y], dim=1)
        x1 = self.ppf(x1)

        xl = self.local_att(x1)
        xg = self.global_att(x1)
        xlg = xl*xg
        wei = self.sigmoid(xlg)
        xo  = wei*x1
        return xo


class PyramidPoolingFusion(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6], reduction_ratio=4):
        """
        金字塔池化特征融合模块
        :param in_channels: 输入特征图的通道数
        :param pool_sizes: 池化层级列表，默认[1,2,3,6]
        :param reduction_ratio: 通道压缩比例，默认4
        """
        super().__init__()
        self.stages = nn.ModuleList()

        # 每个池化层对应一个处理分支
        for size in pool_sizes:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=size),
                    nn.Conv2d(in_channels, in_channels // reduction_ratio,
                              kernel_size=1, bias=False),
                    nn.BatchNorm2d(in_channels // reduction_ratio),
                    nn.ReLU(inplace=True)
                )
            )

        # 后处理卷积
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        :param x: 输入特征图 (B, C, H, W)
        :return: 融合后的特征图 (B, C, H, W)
        """
        h, w = x.size()[2:]
        pyramid_features = [x]  # 保留原始特征

        # 生成多尺度特征
        for stage in self.stages:
            pooled = stage(x)  # (B, C//r, S, S)

            # 双线性插值上采样到原尺寸
            upsampled = F.interpolate(
                pooled, size=(h, w),
                mode='bilinear',
                align_corners=False
            )
            pyramid_features.append(upsampled)

        # 沿通道维度拼接
        fused = torch.cat(pyramid_features, dim=1)  # (B, C*(1+1/r), H, W)

        # 通道压缩
        return self.final_conv(fused)

#高效注意力解码器
class SpatialStripAttention(nn.Module):
    """基于条带池化的空间注意力"""

    def __init__(self, in_channels):
        super().__init__()
        self.conv_h = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_w = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.attn_conv = nn.Sequential(
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 水平条带池化 (B,C,H,1)
        h_pool = F.adaptive_avg_pool2d(x, (None, 1))
        h_feat = self.conv_h(h_pool)

        # 垂直条带池化 (B,C,1,W)
        w_pool = F.adaptive_avg_pool2d(x, (1, None))
        w_feat = self.conv_w(w_pool)

        # 注意力生成
        attn = self.attn_conv(h_feat + w_feat)
        return x * attn


class ECAAttention(nn.Module):
    """高效通道注意力 (ECANet)"""

    def __init__(self, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # (B,C,1,1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B,1,C)
        y = self.conv(y)  # (B,1,C)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B,C,1,1)
        return x * y


class MultiScaleDilationConv(nn.Module):
    """多尺度膨胀卷积并行结构"""

    def __init__(self, channels, dilations=[1, 2, 4]):
        super().__init__()
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3,
                              padding=dilation, dilation=dilation),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.conv_fuse = nn.Conv2d(len(dilations) * channels, channels, 1)

    def forward(self, x):
        branch_outs = [branch(x) for branch in self.branches]
        fused = self.conv_fuse(torch.cat(branch_outs, dim=1))
        return fused


class MSEAD(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        # 空间注意力
        self.spatial_attn = SpatialStripAttention(in_channels)
        # 通道注意力
        self.channel_attn = ECAAttention()
        # 通道压缩
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.ReLU(inplace=True)
        )
        # 多尺度融合
        self.dilation_conv = MultiScaleDilationConv(in_channels // reduction_ratio)
        # 通道恢复
        self.expand = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1)

    def forward(self, x):
        identity = x
        # 注意力双路径
        x = self.spatial_attn(x)  # 空间增强
        x = self.channel_attn(x)  # 通道增强
        # 多尺度处理
        x = self.compress(x)  # 通道压缩
        x = self.dilation_conv(x)  # 多尺度特征融合
        x = self.expand(x)  # 通道恢复
        # 残差连接
        return F.relu(identity + x)





