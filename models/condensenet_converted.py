from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
import math
from layers import ShuffleLayer, ResNet, Conv, CondenseConv, CondenseLinear

import sys
sys.path.append("..")
from main import LTDN

__all__ = ['CondenseNet']

print(f'LTDN: {LTDN}')

class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, args):
        super(_DenseLayer, self).__init__()
        self.group_1x1 = args.group_1x1
        self.group_3x3 = args.group_3x3
        ### 1x1 conv i --> b*k
        self.conv_1 = CondenseConv(in_channels, args.bottleneck * growth_rate,
                                   kernel_size=1, groups=self.group_1x1)
        ### 3x3 conv b*k-->k
        self.conv_2 = Conv(args.bottleneck * growth_rate, growth_rate,
                           kernel_size=3, padding=1, groups=self.group_3x3)

    def forward(self, x):
        x_ = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        return torch.cat([x_, x], 1)

class _DenseLayerLTDN(nn.Module):
    def __init__(self, in_channels, growth_rate, args):
        super(_DenseLayerLTDN, self).__init__()
        
        self.group_1x1 = args.group_1x1
        self.group_3x3 = args.group_3x3
        # upper case
        ### 1x1 conv i --> b*k
        self.conv_1 = CondenseConv(int(in_channels/2), int(args.bottleneck * growth_rate/2),
                                       kernel_size=1, groups=self.group_1x1)
        ### 3x3 conv b*k --> k
        self.conv_2 = Conv(int(args.bottleneck*growth_rate/2), int(growth_rate/2),
                           kernel_size=3, padding=1, groups=self.group_3x3)
        
        # lower case
        ### 1x1 conv i --> b*k
        self.conv_3 = CondenseConv(int(in_channels/2), int(args.bottleneck * growth_rate/2),
                                       kernel_size=1, groups=self.group_1x1)
        ### 3x3 conv b*k --> k
        self.conv_4 = Conv(int(args.bottleneck * growth_rate/2), int(growth_rate/2),
                           kernel_size=3, padding=1, groups=self.group_3x3)

    def forward(self, x):
        input_channels = int(x.shape[1])
        half_input_channels = int(input_channels/2)
        # output_channels = int(x.shape[1])
        # half_output_channels = int(output_channels/2)
        
        upper_input = x[:,0:half_input_channels,:,:] #1, 8, 32, 32
        upper = self.conv_1(upper_input) # 1, 16, 32, 32
        upper = self.conv_2(upper) # 1, 4, 32, 32
        
        lower_input = x[:,half_input_channels:input_channels,:,:] #1,8,32,32
        lower = self.conv_3(lower_input) #1, 16, 32, 32
        lower = self.conv_4(lower) #1, 4, 32, 32
        
        return torch.cat([upper_input, lower, lower_input, upper], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, args):
        super(_DenseBlock, self).__init__()
        if LTDN:
            for i in range(num_layers):
                layer = _DenseLayerLTDN(in_channels + i * growth_rate, growth_rate, args)
                # print(f"layer: \n {layer}")
                self.add_module('denselayer_%d' % (i + 1), layer)
            
        else:
            for i in range(num_layers):
                layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, args)
                # print(f"layer: \n {layer}")
                self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self, in_channels, args):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class CondenseNet(nn.Module):
    def __init__(self, args):

        super(CondenseNet, self).__init__()

        self.stages = args.stages
        self.growth = args.growth
        assert len(self.stages) == len(self.growth)
        self.args = args
        self.progress = 0.0
        if args.data in ['cifar10', 'cifar100']:
            self.init_stride = 1
            self.pool_size = 8
        else:
            self.init_stride = 2
            self.pool_size = 7

        self.features = nn.Sequential()
        ### Initial nChannels should be 3
        self.num_features = 2 * self.growth[0]
        ### Dense-block 1 (224x224)
        self.features.add_module('init_conv', nn.Conv2d(3, self.num_features,
                                                        kernel_size=3,
                                                        stride=self.init_stride,
                                                        padding=1,
                                                        bias=False))
        
        if LTDN:
            resnet = ResNet(int(self.num_features/2), int(self.num_features/2),
                           kernel_size=[1,3,1])
            self.features.add_module('resnet', resnet)
            
        for i in range(len(self.stages)):
            ### Dense-block i
            self.add_block(i)
        
        ### Linear layer
        self.classifier = CondenseLinear(self.num_features, args.num_classes,
                                         0.5)
        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def add_block(self, i):
        ### Check if ith is the last one
        last = (i == len(self.stages) - 1)
        block = _DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            args=self.args,
        )
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            trans = _Transition(in_channels=self.num_features,
                                args=self.args)
            self.features.add_module('transition_%d' % (i + 1), trans)
        else:
            self.features.add_module('norm_last',
                                     nn.BatchNorm2d(self.num_features))
            self.features.add_module('relu_last',
                                     nn.ReLU(inplace=True))
            self.features.add_module('pool_last',
                                     nn.AvgPool2d(self.pool_size))

    def forward(self, x, progress=None):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out
