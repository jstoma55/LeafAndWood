import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import Dataset
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF

def count_parameters(model):
    table = [['module', 'params']]
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.append([name,params])
        total_params+=params
    df = pd.DataFrame(table[1:], columns=table[0])
    print(df)
    print(f"Total Trainable Params: {total_params}")
    return total_params
 
class ConvBlock(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__(3)
        self.block = nn.Sequential(
            ME.MinkowskiBatchNorm(in_channels),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=kernel_size, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=kernel_size, stride=1, dimension=3)
        )
    def forward(self, x):
        return self.block(x)

class EncBlock(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels):
        super(EncBlock, self).__init__(3)
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=3)
        self.pool = ME.MinkowskiMaxPooling(2, stride=2, dimension=3)
    def forward(self, x):
        cat = self.conv(x)
        return cat, self.pool(cat)
    
class DecBlock(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels):
        super(DecBlock, self).__init__(3)
        self.tc = ME.MinkowskiConvolutionTranspose(in_channels, out_channels, kernel_size=2, stride=2, dimension=3)
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=3)
    def forward(self, inputs):
        cat, x = inputs
        o = self.tc(x)
        o = ME.cat(cat, o)
        return self.conv(o)
    
class Seg1(ME.MinkowskiNetwork):
    def __init__(self, in_channels, channel, depth, n_classes, D=3):
        super(Seg1, self).__init__(D)
        channels = [channel*(2**i) for i in range(depth+1)]
        self.enc1 = EncBlock(in_channels, channels[0])
        self.enc2 = EncBlock(channels[0], channels[1])
        self.c1 = ConvBlock(channels[1], channels[2])
        
        self.dec1 = DecBlock(channels[2], channels[1])
        self.dec2 = DecBlock(channels[1], channels[0])
        
        self.out = ME.MinkowskiConvolution(channels[0], n_classes, kernel_size=1, dimension=3)
    def forward(self, x):
        cat1, o = self.enc1(x)
        cat2, o = self.enc2(o)
        o = self.c1(o)
        o = self.dec1((cat2, o))
        o = self.dec2((cat1, o))
        o = self.out(o)
        return o

class Seg2(ME.MinkowskiNetwork):
    def __init__(self, in_channels, channel, depth, n_classes, D=3):
        super(Seg2, self).__init__(D)
        channels = [channel*(2**i) for i in range(depth+1)]
        self.enc1 = EncBlock(in_channels, channels[0])
        self.enc2 = EncBlock(channels[0], channels[1])
        self.enc3 = EncBlock(channels[1], channels[2])
        self.c1 = ConvBlock(channels[2], channels[3])
        
        self.dec1 = DecBlock(channels[3], channels[2])
        self.dec2 = DecBlock(channels[2], channels[1])
        self.dec3 = DecBlock(channels[1], channels[0])
        self.out = ME.MinkowskiConvolution(channels[0], n_classes, kernel_size=1, dimension=3)
    def forward(self, x):
        cat1, o = self.enc1(x)
        cat2, o = self.enc2(o)
        cat3, o = self.enc3(o)
        o = self.c1(o)
        o = self.dec1((cat3, o))
        o = self.dec2((cat2, o))
        o = self.dec3((cat1, o))
        o = self.out(o)
        return o
    
class Seg3(ME.MinkowskiNetwork):
    def __init__(self, in_channels, channel, depth, n_classes, D=3):
        super(Seg3, self).__init__(D)
        channels = [channel*(2**i) for i in range(depth+1)]
        self.enc1 = EncBlock(in_channels, channels[0])
        self.enc2 = EncBlock(channels[0], channels[1])
        self.enc3 = EncBlock(channels[1], channels[2])
        self.enc4 = EncBlock(channels[2], channels[3])
        self.c1 = ConvBlock(channels[3], channels[4])
        
        self.dec1 = DecBlock(channels[4], channels[3])
        self.dec2 = DecBlock(channels[3], channels[2])
        self.dec3 = DecBlock(channels[2], channels[1])
        self.dec4 = DecBlock(channels[1], channels[0])
        self.out = ME.MinkowskiConvolution(channels[0], n_classes, kernel_size=1, dimension=3)
    def forward(self, x):
        cat1, o = self.enc1(x)
        cat2, o = self.enc2(o)
        cat3, o = self.enc3(o)
        cat4, o = self.enc4(o)
        o = self.c1(o)
        o = self.dec1((cat4, o))
        o = self.dec2((cat3, o))
        o = self.dec3((cat2, o))
        o = self.dec4((cat1, o))
        o = self.out(o)
        return o

# Still not finished debugging. Please excuse the name, couldn't think of anything better.
class FrankenSeg(ME.MinkowskiNetwork):
    def __init__(self, in_channels, init_channels, depth, n_classes):
        super(FrankenSeg, self).__init__(3)
        self.depth = depth
        channels = [in_channels]+[init_channels*(2**i) for i in range(depth+1)]
        modules = [EncBlock(channels[i], channels[i+1]) for i in range(depth)]
        modules += [ConvBlock(channels[depth], channels[depth+1])]
        modules += [DecBlock(channels[depth-i+1], channels[depth-i]) for i in range(depth)]
        modules += [ME.MinkowskiConvolution(channels[1], n_classes, kernel_size=1, dimension=3)]
        self.modules = modules
        self.model = nn.ModuleList(modules)
    def forward(self, x):
        shortcut = []
        o = x
        for i in range(self.depth):
            o, sc = self.modules[i](o)
            shortcut.append(sc)
        o = self.modules[self.depth](o)
        shortcut.reverse()
        for i in range(self.depth):
            o = self.modules[self.depth+i+1]((shortcut[i], o))
        o = self.modules[-1](o)
        return o
    