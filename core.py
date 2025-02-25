import torch
import torch.nn as nn
import math
from torchvision.ops import nms
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1, deformable_groups=1):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()


class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    def forward_fuse(self, x):
        return self.act(self.conv(x))

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale
    
class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out



class FA(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, g=1, k=(3, 3), e=0.5):
        super(FA, self).__init__()
        c = int(out_channels * e)
        self.conv1 = Conv(in_channels, c, k=k[0])
        self.dcn = DCNv2(c, c, k[1], 1, groups=g)
        self.triplet_attention = TripletAttention()
        self.conv2 = Conv(c, out_channels, k=1)
        self.use_shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.dcn(out)
        out = self.triplet_attention(out)
        out = self.conv2(out)
        if self.use_shortcut:
            out += identity
        return out


class DSAF(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, g=1, e=0.5):
        super(DSAF, self).__init__()
        self.c = int(out_channels * e)
        self.cv1 = Conv(in_channels, (n + 1) * self.c, 1, 1)
        self.cv2 = Conv((n + 1) * self.c, out_channels, 1, 1)
        self.m = nn.ModuleList([FA(self.c, self.c, shortcut, g, k=(3, 3)) for _ in range(n)])
        self.shortcut = Conv(in_channels, (n + 1) * self.c, 1, 1) if in_channels != (n + 1) * self.c else nn.Identity()

    def forward(self, x):
        y = list(self.cv1(x).chunk(len(self.m) + 1, 1))
        shortcut = self.shortcut(x)

        outputs = [y[0] + shortcut[:, :y[0].size(1), :, :]]  # Make sure shortcut matches the size

        for i, m in enumerate(self.m):
            y[i + 1] = y[i + 1] + shortcut[:, y[i + 1].size(1) * i: y[i + 1].size(1) * (i + 1), :,
                                  :]
            out = m(y[i + 1])
            y[i + 1] = out + y[i + 1]
            outputs.append(y[i + 1])

        out = self.cv2(torch.cat(outputs, 1))
        return out


class SD(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super(SD, self).__init__()
        self.block_size = 2
        c1 = c1 * 4
        self.depthwise_conv = nn.Conv2d(c1, c1, k, s, autopad(k, p, d), groups=c1, dilation=d, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.pointwise_conv = nn.Conv2d(c1, c2, 1, 1, padding=0)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        new_channels = channels * (self.block_size ** 2)
        new_height = height // self.block_size
        new_width = width // self.block_size
        x = x.unfold(2, self.block_size, self.block_size).unfold(3, self.block_size, self.block_size)
        x = x.contiguous().view(batch_size, channels, new_height, new_width, -1)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(batch_size, new_channels, new_height, new_width)
        x = self.bn1(self.depthwise_conv(x))
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        x = x+avg_pool+max_pool
        x = self.bn2(self.pointwise_conv(x))
        return self.act(x)
