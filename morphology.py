import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def padding_shape(kernel_size):
    kernel_size_effective = kernel_size
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return (pad_beg, pad_end, pad_beg, pad_end)


class Morphology(nn.Module):
    '''
    Base class for morpholigical operators
    only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, limit_val,
                 optype, soft_max=True, beta=15):
        '''
        in_channels: scalar
        out_channels: scalar
        kernel_size: scalar
        soft_max: bool, using the soft max rather the torch.max()
        beta: scalar, used by soft_max.
        optype: str, dilation2d or erosion2d.
        '''
        super(Morphology, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.soft_max = soft_max
        self.beta = beta
        self.optype = optype
        self.limit_val = limit_val

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size))

        # init as it is done by default in pytorch
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)

        # compute padding shape once and use everytime
        self.padding_shape = padding_shape(kernel_size)

    def forward(self, x):
        '''
        x: tensor of shape (B,C,H,W)
        '''

        _, _, H, W = x.shape
        
        # padding
        x = F.pad(x, self.padding_shape, mode='constant', value=self.limit_val)
        
        # unfold
        x = self.unfold(x)  # (B, Cin*kH*kW, L), L is the numbers of patches
        x = x.unsqueeze(1)  # (B, 1, Cin*kH*kW, L)

        # the input image may not be square
        # L = x.size(-1)
        # L_sqrt = int(math.sqrt(L))
        # not required using same padding, so just use the original size

        # erosion
        weight = self.weight.view(self.out_channels, -1)  # (Cout, Cin*kH*kW)
        weight = weight.unsqueeze(0).unsqueeze(-1)  # (1, Cout, Cin*kH*kW, 1)

        if self.optype == 'erosion2d':
            x = weight - x  # (B, Cout, Cin*kH*kW, L)
        elif self.optype == 'dilation2d':
            x = weight + x  # (B, Cout, Cin*kH*kW, L)
        else:
            raise ValueError("type should be either erosion2d or dilation2d, \
                             but found {}".format(self.optype))

        if not self.soft_max:
            x, _ = torch.max(x, dim=2, keepdim=False)  # (B, Cout, L)
        else:
            x = torch.logsumexp(x*self.beta, dim=2,
                                keepdim=False) / self.beta  # (B, Cout, L)

        if self.optype == 'erosion2d':
            x = -1 * x

        # instead of fold, we use view to avoid copy
        x = x.view(-1, self.out_channels, H, W)  # (B, Cout, L/2, L/2)

        return x


class Dilation2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size,
                 limit_val=float('-Inf'), soft_max=True, beta=20):
        super(Dilation2d, self).__init__(in_channels, out_channels,
                                         kernel_size, limit_val,
                                         'dilation2d', soft_max, beta,)


class Erosion2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size,
                 limit_val=float('Inf'), soft_max=True, beta=20):
        super(Erosion2d, self).__init__(in_channels, out_channels,
                                        kernel_size, limit_val,
                                        'erosion2d', soft_max, beta)


if __name__ == '__main__':
    # test
    x = torch.randn(2, 3, 16, 16)
    e = Erosion2d(3, 4, 3, soft_max=False)
    y = e(x)
