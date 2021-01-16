from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import Tensor


class Transitional_Layer(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(Transitional_Layer, self).__init__()
        self.add_module('bn',nn.BatchNorm2d(input_dim)),
        self.add_module('relu',nn.ReLU(inplace=True)),
        #this downsamples from the input to output dims
        self.add_module('conv',nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1, padding=1,bias=False)),
        self.add_module('pool',nn.AvgPool2d(kernel_size=2, stride=2))

class Dense_Layer(nn.Module):
    def __init__(self, input_dim, output_dim, growth_rate, bn_size, drop_rate):
        super(Dense_Layer, self).__init__()
        #first conv
        self.add_module('bn', nn.BatchNorm2d(input_dim)),
        self.add_module('relu',nn.ReLU(inplace=True))
        self.add_module('conv',nn.Conv2d(input_dim,bn_size*growth_rate, kernel_size=1, stride=1))
        #second conv
        self.add_module('bn1', nn.BatchNorm2d(bn_size*growth_rate)),
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(bn_size*growth_rate, growth_rate, kernel_size=1, stride=1))
        #dropout
        self.drop_rate = float(drop_rate)

    #bottleneck, takes a list of tensors to a tensor
    def bottleneck(self, x):
        concat_features = torch.cat(x, 1)
        return self.conv1(self.relu1(self.norm1(concat_features)))


    def forward(self, x):
        #checks if it is a tensor, if it is then make it a list to prepare for bottleneck
        if isinstance(x, Tensor):
            prev_features = [x]
        else:
            prev_features = x

        bottleneck_output = self.bottleneck(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        #allows for dropout
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

class Dense_Block(nn.Module):
    def __init__(self, num_layers, input_dim, bn_size, growth_rate, drop_rate):
        super(Dense_Block, self).__init__()
        for i in range(num_layers):
            layer = Dense_Layer(input_dim + i*growth_rate,
                                growth_rate=growth_rate,
                                bn_size=bn_size,
                                drop_rate=drop_rate)
            self.add_module('dense_layer_%d' % (i+1), layer)

    def forward(self, x):
        feats = [x]
        for name, layer in self.items():
            new_features = layer(feats)
            feats.append(new_features)
        return torch.cat(feats,1)

class Dense_Net(nn.Module):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(Dense_Net, self).__init__()
        #conv + avg pool
        self.features = nn.Sequential(OrderedDict([
            ('first_conv',nn.Conv2d(3, init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('first_norm',nn.BatchNorm2d(init_features)),
            ('first_relu',nn.ReLU(inplace=True)),
            ('first_pool',nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

        ]))
        #add the dense blocks which consist of the dense layers
        num_feat=init_features
        for i,num_layers in enumerate(block_config):
            block = Dense_Block(
                num_layers=num_layers,
                input_dim=num_feat,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module('dense_block_%d' % (i+1), block)

            num_feat = num_feat + num_layers * growth_rate
            #if its not the last layer then add a transition layer between dense blocks
            if i !=len(block_config)-1:
                transition = Transitional_Layer(input_dim=num_feat,output_dim=num_feat//2)
                self.features.add_module('transitional_layer_%d' % (i+1), transition)
                num_feat = num_feat//2

        self.features.add_module('last_norm', nn.BatchNorm2d(num_feat))
        self.classifier = nn.Linear(num_feat,num_classes)

        #Init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features,inplace=True)
        out = F.adaptive_avg_pool2d(out, (1,1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out