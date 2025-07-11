import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import IGAB, Illumination_Estimator, TransformerBlock, MSPABlock, MSELA, MSAFB, MSEAD, \
    EfficientTransformerBranch


#创新点1：端到端的低亮度语义分割网络（区别于先增强后分割的思路）
#创新点2：引入transfomer模块到亮度语义分割网络
#创新点3：交叉注意力机制利用亮度特征引导反射率修复和语义分割
class ChannelCompactor(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // reduction, 1),
            nn.BatchNorm2d(out_ch // reduction),
            nn.ReLU(),
            nn.Conv2d(out_ch // reduction, out_ch, 1)
        )

    def forward(self, x):
        return self.conv(x)


#step1：分解网络
class SpatialAttention(nn.Module):
    """空间注意力机制，用于保持光照平滑性"""

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True)
        )
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        # 通道压缩
        x_reduced = self.channel_reduction(x)  # [B, C/r, H, W]

        # 空间注意力生成
        avg_pool = torch.mean(x_reduced, dim=1, keepdim=True)  # [B,1,H,W]
        max_pool, _ = torch.max(x_reduced, dim=1, keepdim=True)  # [B,1,H,W]
        concat = torch.cat([avg_pool, max_pool], dim=1)  # [B,2,H,W]
        spatial_attn = self.spatial_conv(concat)  # [B,1,H,W]
        spatial_attn = torch.sigmoid(spatial_attn)

        # 应用注意力（保持低频平滑）
        return x * spatial_attn.expand_as(x)
#创新点1：多尺度残差模块MSPABlock
#创新点2：引入TDB
class ReflectionBranch(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            MSPABlock(64,64),
            MSPABlock(64,64),
            MSPABlock(64,64),
            nn.ReLU(),

        )
        self.down1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            MSPABlock(128, 128),
            nn.ReLU(),
        )
        self.down2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)


        self.enc3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            MSPABlock(256, 256),
            nn.ReLU(),
        )

        # 解码器
        self.up1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            MSEAD(128)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            MSEAD(128)
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            MSEAD(64)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            MSEAD(64)
        )

        self.final = nn.Conv2d(64, 3, 3, padding=1)

        self.conv1 = nn.Conv2d(in_channels, 64, 1)
        self.trans_1 = TransformerBlock(64)
        self.conv2 = nn.Conv2d(64, 128, 1)
        self.trans_2 = TransformerBlock(128)
        self.conv3 = nn.Conv2d(128, 64, 1)
        self.trans_3 = TransformerBlock(64)
        self.conv4 = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        #双分支网络
        e1 = self.enc1(x)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        e3 = self.enc3(d2)

        # 解码过程
        u1 = self.up1(e3)
        c1 = torch.cat([u1, e2], dim=1)
        d1 = self.dec1(c1)
        u2 = self.up2(d1)
        c2 = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(c2)
        conv_r = self.final(d2)

        #transfomer分支
        t1 = self.conv1(x)
        t1 = self.trans_1(t1)
        t2 = self.conv2(t1)
        t2 = self.trans_2(t2)
        t3 = self.conv3(t2)
        t3 = self.trans_3(t3)
        t4 = self.conv4(t3)
        r = conv_r+t4
        r = torch.sigmoid(r)
        return r


#创新点1：SpatialAttention帮助平滑
class IlluminationBranch(nn.Module):
    """照度分支（无跳跃连接的类UNet结构）"""

    def __init__(self, in_channels=3):
        super().__init__()
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )

        # 解码器
        self.up1 = nn.Sequential(nn.Conv2d(128, 64, 3, stride=1, padding=1),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.Sequential(nn.Conv2d(64, 32, 3, stride=1, padding=1),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            SpatialAttention(32),
        )

        self.final = nn.Conv2d(32, 1, 1)  # 输出1通道照度图

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)  # [B,32,H,W]
        e2 = self.enc2(self.down1(e1))  # [B,64,H/2,W/2]
        e3 = self.enc3(self.down2(e2))  # [B,128,H/4,W/4]

        # 解码（无跳跃连接）
        d1 = self.dec1(self.up1(e3))  # [B,64,H/2,W/2]
        d2 = self.dec2(self.up2(d1))  # [B,32,H,W]

        return torch.sigmoid(self.final(d2))  # [B,1,H,W]

class DECOMP_net(nn.Module):
    """完整的Retinex分解网络"""
    def __init__(self):
        super().__init__()
        self.reflection_branch = ReflectionBranch()
        self.illumination_branch = IlluminationBranch()

    def forward(self, x):
        R = self.reflection_branch(x)  # 反射率分量 [B,3,H,W]
        L = self.illumination_branch(x)  # 照度分量 [B,1,H,W]
        return L,R

#step2:反射率修复和语义分割
#创新点1：多尺度高效空间注意力编码器（MSELA）
#创新点2：亮度特征引导的反射率修复和语义分割（IGAB)
class Seg_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 6
        self.ill = Illumination_Estimator(n_fea_middle=32)

        # Encoder layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            MSELA(32),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            MSELA(64),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            MSELA(128),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            MSELA(256),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            MSELA(256),
        )

        # 同步卷积
        self.conv1_illfea = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv2_illfea  = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv3_illfea  = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv4_illfea  = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv5_illfea  = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.igab_1 =IGAB(32)
        self.igab_2 = IGAB(64)
        self.igab_3 = IGAB(128)
        self.igab_4 = IGAB(256)
        self.igab_5 = IGAB(256)

        # decoder
        self.mid_r = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(), )

        self.mid_s = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(), )
        self.mid_l = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(), )

        self.deconv4_r = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.deconv3_r = nn.Sequential(
            ChannelCompactor(768, 128),
            # nn.Conv2d(768, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.deconv2_r = nn.Sequential(
            ChannelCompactor(384, 64),
            # nn.Conv2d(384, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.deconv1_r = nn.Sequential(
            ChannelCompactor(192, 32),
            # nn.Conv2d(192, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
        self.iagb_r4 = IGAB(512)
        self.iagb_r3 = IGAB(768)
        self.iagb_r2 = IGAB(384)
        self.iagb_r1 = IGAB(192)

        # 同步卷积
        self.deconv4_illfea = nn.Sequential(
            nn.Conv2d(512, 768, 1, 1, 0),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),

        )

        self.deconv3_illfea = nn.Sequential(
            nn.Conv2d(768, 384, 1, 1, 0),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.deconv2_illfea = nn.Sequential(
            nn.Conv2d(384, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.deconv4_s = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.deconv3_s = nn.Sequential(
            ChannelCompactor(768,128),
            # nn.Conv2d(768, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.deconv2_s = nn.Sequential(
            ChannelCompactor(384, 64),
            # nn.Conv2d(384, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.deconv1_s = nn.Sequential(
            ChannelCompactor(192, 32),
            # nn.Conv2d(192, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

        self.iagb_s4 = IGAB(512)
        self.iagb_s3 = IGAB(768)
        self.iagb_s2 = IGAB(384)
        self.iagb_s1 = IGAB(192)

        self.out_r = nn.Sequential(
            nn.Conv2d(96, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
        )

        self.out_s = nn.Sequential(
            nn.Conv2d(96, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, r,l):
        illu_fea, illu_map = self.ill(r,l)
        x1 = self.conv1(torch.cat((r, illu_map), 1))

        x1_igab = self.igab_1(x1,illu_fea)
        # 16 x 240 x 320
        x2 = self.conv2(x1_igab)

        illu_fea = self.conv2_illfea(illu_fea)#64
        x2_igab = self.igab_2(x2,illu_fea)
        # 32 x 120 x 160
        x3 = self.conv3(x2_igab)

        illu_fea = self.conv3_illfea(illu_fea)#128
        x3_igab = self.igab_3(x3,illu_fea)
        # 64 x 60 x 80
        x4 = self.conv4(x3_igab)
        illu_fea = self.conv4_illfea(illu_fea)#256
        x4_igab = self.igab_4(x4,illu_fea)
        # 128 x 30 x 40
        x5 = self.conv5(x4_igab)

        illu_fea = self.conv5_illfea(illu_fea)
        x5_igab = self.igab_5(x5,illu_fea)
        # 256 x 15 x 20

        xmid_r = torch.cat((self.mid_r(x5_igab), x5_igab), dim=1)
        xmid_s = torch.cat((self.mid_s(x5_igab), x5_igab), dim=1)
        # 512 x 15 x 20
        illu_fea = torch.cat((self.mid_l(illu_fea), illu_fea), dim=1)
        xmid_r_igab = self.iagb_r4(xmid_r,illu_fea)
        x4d_r = self.deconv4_r(xmid_r_igab)
        xmid_s_igab = self.iagb_s4(xmid_s,illu_fea)
        x4d_s = self.deconv4_s(xmid_s_igab)


        concat_3_r= torch.cat([x4d_r] + [x4d_s] + [x4], dim=1)#1,768,32,32
        illu_fea = self.deconv4_illfea(illu_fea)
        concat_3_r_igab = self.iagb_r3(concat_3_r,illu_fea)#1,512,56,56
        x3d_r = self.deconv3_r(concat_3_r_igab)
        concat_3_s = torch.cat([x4d_s] + [x4d_r] + [x4], dim=1)
        concat_3_s_igab = self.iagb_s3(concat_3_s,illu_fea)
        x3d_s = self.deconv3_s(concat_3_s_igab)


        concat_2_r= torch.cat([x3d_r] + [x3d_s] + [x3], dim=1)
        illu_fea = self.deconv3_illfea(illu_fea)
        concat_2_r_igab = self.iagb_r2(concat_2_r,illu_fea)
        x2d_r = self.deconv2_r(concat_2_r_igab)
        concat_2_s = torch.cat([x3d_s] + [x3d_r] + [x3], dim=1)
        concat_2_s_igab = self.iagb_s2(concat_2_s,illu_fea)
        x2d_s = self.deconv2_s(concat_2_s_igab)


        concat_1_r= torch.cat([x2d_r] + [x2d_s] + [x2], dim=1)
        illu_fea =self.deconv2_illfea(illu_fea)
        concat_1_r_igab = self.iagb_r1(concat_1_r,illu_fea)
        x1d_r = self.deconv1_r(concat_1_r_igab)
        concat_1_s = torch.cat([x2d_s] + [x2d_r] + [x2], dim=1)
        concat_1_s_igab = self.iagb_s1(concat_1_s,illu_fea)
        x1d_s = self.deconv1_s(concat_1_s_igab)

        out_r = torch.sigmoid(self.out_r(torch.cat([x1d_r] + [x1d_s] + [x1], dim=1)))
        out_s = self.out_s(torch.cat([x1d_s] + [x1d_r] + [x1], dim=1))
        return out_r, out_s

