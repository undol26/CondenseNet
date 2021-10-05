from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from layers import ResNet, Conv, LearnedGroupConv


__all__ = ['DenseNet_LGC']


def make_divisible(x, y):
    return int((x // y + 1) * y if x % y else x)


class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, args):
        super(_DenseLayer, self).__init__()
        self.group_1x1 = args.group_1x1
        self.group_3x3 = args.group_3x3
        ### 1x1 conv i --> b*k
        self.conv_1 = LearnedGroupConv(in_channels, args.bottleneck * growth_rate,
                                       kernel_size=1, groups=self.group_1x1,
                                       condense_factor=args.condense_factor,
                                       dropout_rate=args.dropout_rate)
        ### 3x3 conv b*k-->k
        self.conv_2 = Conv(args.bottleneck * growth_rate, growth_rate,
                           kernel_size=3, padding=1, groups=self.group_3x3)

    def forward(self, x):
        x_ = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        return torch.cat([x_, x], 1)

class _DenseLayerLTDN(nn.Module):
    def __init__(self, in_channels, growth_rate, path, args):
        super(_DenseLayerLTDN, self).__init__()
        
        self.group_1x1 = args.group_1x1
        self.group_3x3 = args.group_3x3
        self.path = path
        
        for i in range(path):
            ### 1x1 conv i --> b*k
            layer1 = LearnedGroupConv(int(in_channels/path), int(args.bottleneck * growth_rate/path),
                                        kernel_size=1, groups=self.group_1x1,
                                        condense_factor=args.condense_factor,
                                        dropout_rate=args.dropout_rate)
            self.add_module('path_%d%d' %((i + 1),  1), layer1)
            ### 3x3 conv b*k --> k
            layer2 = Conv(int(args.bottleneck * growth_rate/path), int(growth_rate/path),
                            kernel_size=3, padding=1, groups=self.group_3x3)
            self.add_module('path_%d%d' % ((i + 1), 2), layer2)

    def forward(self, x):
        input_channels = int(x.shape[1])
        path = self.path
        num_input_part_channels = int(input_channels/path)
            
        input_part = {}
        output_part = {}
        returnList = []
        for i in range(path):
            temp_input_part = x[:,i*num_input_part_channels:(i+1)*num_input_part_channels,:,:]
            input_part['input_part{0}'.format(i+1)] = temp_input_part
        
            output_part['output_part{0}'.format(i+1)] = eval(f'self.path_{i+1}{1}')(input_part['input_part{0}'.format(i+1)])
            output_part['output_part{0}'.format(i+1)] = eval(f'self.path_{i+1}{2}')(output_part['output_part{0}'.format(i+1)])
            
        for i in range(path):
            if i%2==0:
                returnList.append(input_part['input_part{0}'.format(i+1)])
                returnList.append(output_part['output_part{0}'.format(i+2)])
                returnList.append(input_part['input_part{0}'.format(i+2)])
                returnList.append(output_part['output_part{0}'.format(i+1)])
            
        return torch.cat(returnList, 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, path, args):
        super(_DenseBlock, self).__init__()
        if args.ltdn_model:
            for i in range(num_layers):
                layer = _DenseLayerLTDN(in_channels + i * growth_rate, growth_rate, path, args)
                self.add_module('denselayer_%d' % (i + 1), layer)
            
        else:
            for i in range(num_layers):
                layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, args)
                self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(_Transition, self).__init__()
        self.conv = LearnedGroupConv(in_channels, out_channels,
                                     kernel_size=1, groups=args.group_1x1,
                                     condense_factor=args.condense_factor)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseNet_LGC(nn.Module):
    def __init__(self, args):

        super(DenseNet_LGC, self).__init__()

        self.stages = args.stages
        self.growth = args.growth
        self.paths = args.paths
        self.reduction = args.reduction
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
        ### Set initial width to 2 x growth_rate[0]
        self.num_features = 2 * self.growth[0]
        ### Dense-block 1 (224x224)
        self.features.add_module('init_conv', nn.Conv2d(3, self.num_features,
                                                        kernel_size=3,
                                                        stride=self.init_stride,
                                                        padding=1,
                                                        bias=False))
        
        if args.ltdn_model:
            resnet = ResNet(int(self.num_features/2), int(self.num_features/2),
                        kernel_size=[1,3,1])
            self.features.add_module('resnet', resnet)
        
        for i in range(len(self.stages)):
            ### Dense-block i
            self.add_block(i)
        ### Linear layer
        self.classifier = nn.Linear(self.num_features, args.num_classes)

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
            path=self.paths[i],
            args=self.args
        )
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            out_features = make_divisible(math.ceil(self.num_features * self.reduction),
                                          self.args.group_1x1)
            trans = _Transition(in_channels=self.num_features,
                                out_channels=out_features,
                                args=self.args)
            self.features.add_module('transition_%d' % (i + 1), trans)
            self.num_features = out_features
        else:
            self.features.add_module('norm_last',
                                     nn.BatchNorm2d(self.num_features))
            self.features.add_module('relu_last',
                                     nn.ReLU(inplace=True))
            ### Use adaptive ave pool as global pool
            self.features.add_module('pool_last',
                                     nn.AvgPool2d(self.pool_size))

    def forward(self, x, progress=None):
        if progress:
            LearnedGroupConv.global_progress = progress
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out
