import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class PatchBranch(nn.Module):
    def __init__(self, in_channel=19, out_channel=50, phase='train'):
        """
        the numbers 40, 20, 76 are fixed by now, they are according to the example for GTA5 in the
        implementation guide: https://docs.google.com/document/d/1w235D1vonIl6ER7AEfOOp8T0OFUiLwXCDFUdAra62RU/edit
        """

        super(PatchBranch, self).__init__()

        # data shape: (u, v, in_channel)
        self.adapool = nn.AdaptiveAvgPool2d((40,40))
        # data shape: (40, 40, in_channel)
        self.conv_1 = nn.Conv2d(in_channel, 76, kernel_size=2, stride=2, padding=0)
        self.relu_1 = nn.ReLU()
        # data shape: (20, 20, 76)
        self.conv_2 = nn.Conv2d(76, out_channel, kernel_size=1, stride=1, padding=1)
        self.relu_2 = nn.ReLU()
        # data shape: (20, 20, out_channel)

        self.model = nn.Sequential(
            self.adapool,
            self.conv_1,
            self.relu_1,
            self.conv_2,
            self.relu_2,
        )

    def forward(self, x):
        return self.model(x)


    ### the code below for loss calculation and optimization(learning rate) are copied from ResNet101 in deeplab.py

    def get_1x_lr_params_NOscale(self):

        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):

        b = []
        b.append(self.layer5.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]

    def adjust_learning_rate(self, args, optimizer, i):
        lr = args.learning_rate * ((1 - float(i) / args.num_steps) ** (args.power))
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10

    def CrossEntropy2d(self, predict, target, weight=None, size_average=True):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != 255)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=size_average)
        return loss