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