import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet import *
from thop import profile
from matplotlib import pyplot as plt

class Resblock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Resblock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class NonLocalBlock(nn.Module):
    def __init__(self, planes, reduce_ratio=8):
        super(NonLocalBlock, self).__init__()
        inter_planes = planes // reduce_ratio
        self.query_conv = nn.Conv2d(planes, inter_planes, kernel_size=1)
        self.key_conv = nn.Conv2d(planes, inter_planes, kernel_size=1)
        self.value_conv = nn.Conv2d(planes, planes, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()

        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        proj_query = proj_query.contiguous().view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = proj_key.contiguous().view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = proj_value.contiguous().view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, -1, height, width)

        out = self.gamma * out
        return out


class pixel_attention(nn.Sequential):
    def __init__(self, in_chs):
        super (pixel_attention, self).__init__(
            nn.Conv2d(in_chs, in_chs // 4, kernel_size=1),
            nn.BatchNorm2d(in_chs//4),
            nn.ReLU(True),
            nn.Conv2d(in_chs // 4, in_chs, kernel_size=1),
            nn.BatchNorm2d(in_chs),
            nn.Sigmoid(),
        )

class FE_module1(nn.Module):
    def __init__(self, in_chs, out_chs, reduce_ratio_nl):
        super(FE_module1, self).__init__()
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        self.adpool = nn.AdaptiveAvgPool2d(1)
        self.pool4 = nn.AdaptiveMaxPool2d((1, 1))
        self.pool5 = nn.AdaptiveMaxPool2d((2, 2))
        self.pool6 = nn.AdaptiveMaxPool2d((3, 3))
        self.pool7 = nn.AdaptiveMaxPool2d((6, 6))
        self.non_local_att4 = NonLocalBlock(out_chs, reduce_ratio_nl)
        self.non_local_att5 = NonLocalBlock(out_chs, reduce_ratio_nl)
        self.non_local_att6 = NonLocalBlock(out_chs, reduce_ratio_nl)
        self.non_local_att7 = NonLocalBlock(out_chs, reduce_ratio_nl)
        self.conv_att4 = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=1),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(True),
        )
        self.conv_att5 = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=1),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(True),
        )
        self.conv_att6 = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=1),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(True),
        )
        self.conv_att7 = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=1),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(True),
        )

        self.final4 = nn.Sequential(
            nn.Conv2d(out_chs, out_chs, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(True),
            nn.Conv2d(out_chs, out_chs, kernel_size=1),
            nn.BatchNorm2d(out_chs),
        )

        self.final5 = nn.Sequential(
            nn.Conv2d(out_chs, out_chs, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(True),
            nn.Conv2d(out_chs, out_chs, kernel_size=1),
            nn.BatchNorm2d(out_chs),
        )

        self.final6 = nn.Sequential(
            nn.Conv2d(out_chs, out_chs, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(True),
            nn.Conv2d(out_chs, out_chs, kernel_size=1),
            nn.BatchNorm2d(out_chs),
        )

        self.final7 = nn.Sequential(
            nn.Conv2d(out_chs, out_chs, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(True),
            nn.Conv2d(out_chs, out_chs, kernel_size=1),
            nn.BatchNorm2d(out_chs),
        )


        self.fc = nn.Conv2d(in_chs + 4 * out_chs, in_chs, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_chs)
        self.pa = pixel_attention(in_chs)

    def forward(self, x):
        batch_size, C, height, width = x.size()
        #
        y4 = self.pool4(x)
        y4 = self.conv_att4(y4)
        y4 = F.interpolate(y4, [height, width], mode='bilinear', align_corners=True)
        y4_nonlocal = self.non_local_att4(y4)
        y4_nonlocal = y4_nonlocal + y4

        y5 = self.pool5(x)
        y5 = self.conv_att5(y5)
        y5 = F.interpolate(y5, [height, width], mode='bilinear', align_corners=True)

        y5_nonlocal = self.non_local_att5(y5)
        y5_nonlocal = y5_nonlocal + y5

        y6 = self.pool6(x)
        y6 = self.conv_att6(y6)
        y6 = F.interpolate(y6, [height, width], mode='bilinear', align_corners=True)

        y6_nonlocal = self.non_local_att6(y6)
        y6_nonlocal = y6_nonlocal + y6

        y7 = self.pool7(x)
        y7 = self.conv_att7(y7)
        y7 = F.interpolate(y7, [height, width], mode='bilinear', align_corners=True)

        y7_nonlocal = self.non_local_att7(y7)
        y7_nonlocal = y7_nonlocal + y7

        out = self.relu(self.bn(self.fc(torch.cat([y4_nonlocal, y5_nonlocal, y6_nonlocal, y7_nonlocal, x], 1))))

        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, planes_high, planes_low, planes_out):
        super(FeatureFusionModule, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(planes_low, planes_low // 4, kernel_size=1),
            nn.BatchNorm2d(planes_low // 4),
            nn.ReLU(True),

            nn.Conv2d(planes_low // 4, planes_low, kernel_size=1),
            nn.BatchNorm2d(planes_low),
            nn.Sigmoid(),
        )
        self.plus_conv = nn.Sequential(
            nn.Conv2d(planes_low*3, planes_low, kernel_size=1),
            nn.BatchNorm2d(planes_low),
            nn.ReLU(True)
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes_low, planes_low // 4, kernel_size=1),
            nn.BatchNorm2d(planes_low // 4),
            nn.ReLU(True),

            nn.Conv2d(planes_low // 4, planes_low, kernel_size=1),
            nn.BatchNorm2d(planes_low),
            nn.Sigmoid(),
        )
        self.end_conv = nn.Sequential(
            nn.Conv2d(planes_high, planes_low, 3, 1, 1),
            nn.BatchNorm2d(planes_low),
            nn.ReLU(True),
        )

    def forward(self, x_high, x_low):
        x_high = self.end_conv(x_high)
        feat = x_low + x_high

        pa = self.pa(x_low) * x_high
        ca = self.ca(x_high) * x_low
        feat = self.plus_conv(torch.cat([pa, ca, feat], 1))
        # feat = feat * ca
        # feat = feat * pa
        return feat


class PSP_NonLocal(nn.Module):
    def __init__(self, num_classes, backbone):
        super(PSP_NonLocal, self).__init__()
        self.up = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=True)
        elif backbone == 'resnet34':
            self.backbone = resnet34(pretrained=True)
        else:
            raise NotImplementedError

        self.femodule = FE_module1(256, 64, 4)

        self.FeaFus2 = FeatureFusionModule(256, 128, 128)
        self.FeaFus1 = FeatureFusionModule(128, 64, 64)
        self.final = nn.Sequential(nn.Conv2d(64, num_classes, kernel_size=1))

        for m in self.modules():
            # 也可以判断是否为conv2d，使用相应的初始化方式
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        fig = plt.figure()
        _, _, hei, wid = x.shape

        c0, c1, c2 = self.backbone(x)
        # tmp = c2.squeeze(0).cpu().numpy()
        # for i in range(16):
        #     ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        #     plt.imshow(tmp[i, :, :], cmap="jet")
        # plt.show()# 使用热力图
        asppm_out = self.femodule(c2)

        p2 = asppm_out
        # p3_ = self.up8(self.conv_p3(p2))
        AF2 = self.FeaFus2(self.up(p2), c1)
        p1 = AF2
        AF1 = self.FeaFus1(self.up(p1), c0)
        out = self.final(torch.cat([AF1],1))
        return out


if __name__ == '__main__':
    model = PSP_NonLocal(num_classes=1, backbone='resnet18')
    x = torch.rand(1, 3, 256, 256)
    flops, params = profile(model, inputs=(x,))
    outs = model(x)
    print(outs.size())
