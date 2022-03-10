import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import time


# from .common import ShiftMean


class ResBlock(nn.Module):
    def __init__(self, n_feats, expansion_ratio, res_scale=1.0, low_rank_ratio=0.75):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.gw1 = weight_norm(nn.Conv2d(n_feats, int(n_feats * expansion_ratio * 0.5), kernel_size=1))
        self.gw2 = weight_norm(nn.Conv2d(n_feats, int(n_feats * expansion_ratio * 0.5), kernel_size=3, padding=1))
        self.re = nn.PReLU()
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(n_feats * expansion_ratio, int(n_feats * low_rank_ratio), kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(int(n_feats * low_rank_ratio), n_feats, kernel_size=3, padding=1))
        )

    def forward(self, x):
        gw1 = self.gw1(x)
        gw2 = self.gw2(x)
        gw = torch.cat([gw1, gw2], dim=1)
        gw = self.re(gw)
        return x + self.module(gw) * self.res_scale


class ResBlock1(nn.Module):
    def __init__(self, n_feats, expansion_ratio, res_scale=1.0, low_rank_ratio=0.75):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.gw1 = weight_norm(nn.Conv2d(n_feats, int(n_feats * expansion_ratio), kernel_size=1))
        self.gw2 = weight_norm(nn.Conv2d(int(n_feats * expansion_ratio), int(n_feats * expansion_ratio),
                                         groups=int(n_feats * expansion_ratio), kernel_size=3, padding=1))
        self.gw3 = weight_norm(nn.Conv2d(int(n_feats * expansion_ratio), n_feats, kernel_size=1))
        self.re = nn.PReLU()
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(n_feats * expansion_ratio, int(n_feats * low_rank_ratio), kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(int(n_feats * low_rank_ratio), n_feats, kernel_size=3, padding=1))
        )

    def forward(self, x):
        gw1 = self.gw1(x)
        gw1 = self.re(gw1)
        gw2 = self.gw2(gw1)
        gw2 = self.re(gw2)
        gw3 = self.gw3(gw2)
        gw3 = self.re(gw3)
        return gw3


class Body(nn.Module):
    def __init__(self):
        super(Body, self).__init__()
        w_residual = [ResBlock(42, 6, 1.0, 0.75)
                      for _ in range(12)]
        self.module = nn.Sequential(*w_residual)

    def forward(self, x):
        x1 = self.module(x)
        return torch.cat([x, x1], dim=1)


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv = double_conv(out_ch, out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResBlock_3(nn.Module):
    def __init__(self, n_feats):
        super(ResBlock_3, self).__init__()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = weight_norm(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=2, padding=1))
        self.pool3 = weight_norm(nn.Conv2d(n_feats, n_feats, kernel_size=1))

    def forward(self, x):
        p1 = self.pool1(x)
        p2 = self.pool2(x)
        p3 = self.pool3(p2)
        return p1 + p3


class Body_3(nn.Module):
    def __init__(self):
        super(Body_3, self).__init__()
        self.con = weight_norm(nn.Conv2d(3, 30, kernel_size=3, padding=1))
        w_residual = [ResBlock_3(30)
                      for _ in range(3)]
        self.module = nn.Sequential(*w_residual)

    def forward(self, x):
        x = self.con(x)
        x = self.module(x)
        return x


class DSS(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DSS, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(8)
        self.upsample = nn.Upsample(scale_factor=8, mode='nearest')
        head = [weight_norm(nn.Conv2d(30, 42, kernel_size=3, padding=1))]
        tail1 = [nn.Upsample(scale_factor=4, mode='nearest'),
                 weight_norm(nn.Conv2d(84, 84, kernel_size=1)),
                 nn.PReLU(),
                 weight_norm(nn.Conv2d(84, 84, kernel_size=3, padding=1)),
                 nn.PReLU()]
        boundary = [weight_norm(nn.Conv2d(84, 84, kernel_size=3, padding=1)),
                    nn.PReLU(),
                    weight_norm(nn.Conv2d(84, 84, kernel_size=3, padding=1)),
                    # weight_norm(nn.Conv2d(32, 32, kernel_size=3, padding=1)),
                    weight_norm(nn.Conv2d(84, 1, kernel_size=3, padding=1)),
                    weight_norm(nn.Conv2d(1, 1, kernel_size=3, padding=1))]
        tail2 = [weight_norm(nn.Conv2d(84, 32, kernel_size=3, padding=1)),
                 nn.PReLU(),
                 nn.Conv2d(32, 20, kernel_size=3, padding=1),
                 nn.PReLU(),
                 nn.Upsample(scale_factor=2, mode='nearest'),
                 nn.Conv2d(20, 20, kernel_size=3, padding=1),
                 nn.Conv2d(20, out_channels, kernel_size=3, padding=1)]
        sr_head = [weight_norm(nn.Conv2d(30, 42, kernel_size=3, padding=1)),  # 32
                   nn.PReLU(),
                   weight_norm(nn.Conv2d(42, 64, kernel_size=3, padding=1)),
                   nn.PReLU()]
        sr_body = [ResBlock(64, 6, 1.0, 0.75)
                   for _ in range(5)]
        sr_tail = [weight_norm(nn.Conv2d(64, 192, kernel_size=3, padding=1)),
                   nn.PReLU(),
                   weight_norm(nn.Conv2d(192, 192, kernel_size=3, padding=1)),
                   nn.PixelShuffle(8)]
        # self.pre = nn.Sequential(*pre)
        self.pre = Body_3()
        self.head = nn.Sequential(*head)
        self.body = Body()
        self.tail1 = nn.Sequential(*tail1)
        self.tail2 = nn.Sequential(*tail2)
        self.boundary = nn.Sequential(*boundary)
        # self.skip = nn.Sequential(*skip)
        self.sr_head = nn.Sequential(*sr_head)
        self.sr_body = nn.Sequential(*sr_body)
        self.sr_tail = nn.Sequential(*sr_tail)

    def forward(self, x):
        x = self.pre(x)
        img_sr = self.sr_head(x)
        img_sr = self.sr_body(img_sr)
        img_sr = self.sr_tail(img_sr)
        x = self.head(x)
        x = self.body(x)
        x = self.tail1(x)
        boundary = self.boundary(x)
        x = self.tail2(x)
        return x, img_sr, boundary
