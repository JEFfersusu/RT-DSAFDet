import torch
import torch.nn as nn
# from ultralytics.ops_dcnv3.modules.dcnv3 import DCNv3_pytorch  # pytorch版
import math
from torchvision.ops import nms

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
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
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
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

# class DCNv3_PyTorch(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1, dilation=1):
#         super().__init__()
#         self.conv = Conv(in_channels, out_channels, k=1)
#         self.dcnv3 = DCNv3_pytorch(out_channels, kernel_size=kernel_size, stride=stride, group=groups, dilation=dilation)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.gelu = nn.GELU()

#     def forward(self, x):
#         x = self.conv(x)
#         x = x.permute(0, 2, 3, 1)
#         x = self.dcnv3(x)
#         x = x.permute(0, 3, 1, 2)
#         x = self.gelu(self.bn(x))
#         return x

class MSBlockLayer(nn.Module):
    def __init__(self, inc, ouc, k) -> None:
        super().__init__()
        self.in_conv = Conv(inc, ouc, 1)
        self.mid_conv = Conv(ouc, ouc, k, g=ouc)
        self.out_conv = Conv(ouc, inc, 1)

    def forward(self, x):
        return self.out_conv(self.mid_conv(self.in_conv(x)))

class MSBlock(nn.Module):
    def __init__(self, inc, ouc, kernel_sizes, in_expand_ratio=3., mid_expand_ratio=2., layers_num=3, in_down_ratio=2.) -> None:
        super().__init__()
        in_channel = int(inc * in_expand_ratio // in_down_ratio)
        self.mid_channel = in_channel // len(kernel_sizes)
        groups = int(self.mid_channel * mid_expand_ratio)
        self.in_conv = Conv(inc, in_channel)
        self.mid_convs = []
        for kernel_size in kernel_sizes:
            if kernel_size == 1:
                self.mid_convs.append(nn.Identity())
                continue
            mid_convs = [MSBlockLayer(self.mid_channel, groups, k=kernel_size) for _ in range(int(layers_num))]
            self.mid_convs.append(nn.Sequential(*mid_convs))
        self.mid_convs = nn.ModuleList(self.mid_convs)
        self.out_conv = Conv(in_channel, ouc, 1)
        self.attention = None

    def forward(self, x):
        out = self.in_conv(x)
        channels = []
        for i, mid_conv in enumerate(self.mid_convs):
            channel = out[:, i * self.mid_channel:(i + 1) * self.mid_channel, ...]
            if i >= 1:
                channel = channel + channels[i - 1]
            channel = mid_conv(channel)
            channels.append(channel)
        out = torch.cat(channels, dim=1)
        out = self.out_conv(out)
        if self.attention is not None:
            out = self.attention(out)
        return out

class Bottleneck_Combined(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = DCNv2(c_, c2, k[1], 1, groups=g)
        self.TripleAt = TripletAttention()
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv1(x)
        out = self.cv2(out)
        out = self.TripleAt(out)
        return x + out if self.add else out

class C2f_Combined(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_Combined(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
# class MS_Combined(nn.Module):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         # self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         # self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(MSBlock(self.c, self.c, kernel_sizes=[3, 5, 7]) for _ in range(n))

#     def forward(self, x):
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))

#     def forward_split(self, x):
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))
class FA_Block(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, g=1, k=(3, 3), e=0.5):
        super(FA_Block, self).__init__()
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

class DSAF_Block(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, g=1, e=0.5):
        super(DSAF_Block, self).__init__()
        self.c = int(out_channels * e)
        self.cv1 = Conv(in_channels, (n + 1) * self.c, 1, 1)
        self.cv2 = Conv((n + 1) * self.c, out_channels, 1, 1)
        self.m = nn.ModuleList([FA_Block(self.c, self.c, shortcut, g, k=(3, 3)) for _ in range(n)])
        self.shortcut = Conv(in_channels, (n + 1) * self.c, 1, 1) if in_channels != (n + 1) * self.c else nn.Identity()

    def forward(self, x):
        y = list(self.cv1(x).chunk(len(self.m) + 1, 1))
        shortcut = self.shortcut(x)
        
        outputs = [y[0] + shortcut[:, :y[0].size(1), :, :]]  # Make sure shortcut matches the size

        for i, m in enumerate(self.m):
            y[i + 1] = y[i + 1] + shortcut[:, y[i + 1].size(1) * i: y[i + 1].size(1) * (i + 1), :, :]  # Pre-FA_Block residual connection
            out = m(y[i + 1])
            y[i + 1] = out + y[i + 1]  # Post-FA_Block residual connection
            outputs.append(y[i + 1])

        out = self.cv2(torch.cat(outputs, 1))
        return out
import torch
import torch.nn as nn
 
# __all__ = ['S2DConv']
    
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p    

class S2DConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super(S2DConv, self).__init__()
        self.block_size = 2
        c1 = c1 * 4
        self.depthwise_conv = nn.Conv2d(c1, c1, k, s, autopad(k, p, d), groups=c1, dilation=d, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.pointwise_conv = nn.Conv2d(c1, c2, 1, 1, padding=0)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        new_channels = channels * (self.block_size ** 2)
        new_height = height // self.block_size
        new_width = width // self.block_size
        x = x.unfold(2, self.block_size, self.block_size).unfold(3, self.block_size, self.block_size)
        x = x.contiguous().view(batch_size, channels, new_height, new_width, -1)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(batch_size, new_channels, new_height, new_width)
        return self.act(self.bn2(self.pointwise_conv(self.bn1(self.depthwise_conv(x)))))

