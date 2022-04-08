import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet50

class Downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(y, 2, stride=2)

        return x, y


class Upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample_block, self).__init__()
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        #x = self.up(x)
        #x = torch.cat((x, y), dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        return x

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out

class Unet(nn.Module):
    def __init__(self, num_classes):
        in_chan = 3
        super(Unet, self).__init__()
        self.encoder = resnet50()
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.conv0_0 = DoubleConv(input_channels, 64)

        nb_filter = [64, 256, 512, 1024, 2048]
        self.encoder = resnet50()
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.conv0_0 = DoubleConv(input_channels, 64)

        self.conv0_0 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)
        self.maxpool = self.encoder.maxpool
        self.conv1_0 = self.encoder.layer1
        self.conv2_0 = self.encoder.layer2
        self.conv3_0 = self.encoder.layer3
        self.conv4_0 = self.encoder.layer4
        # self.down1 = Downsample_block(3, nb_filter[0])  # ResBlock
        # self.down2 = Downsample_block(nb_filter[0], nb_filter[1])
        # self.down3 = Downsample_block(nb_filter[1], nb_filter[2])
        # self.down4 = Downsample_block(nb_filter[2], nb_filter[3])
        # self.conv1 = nn.Conv2d(nb_filter[3], nb_filter[4], 3, padding=1)
        # self.bn1 = nn.BatchNorm2d(nb_filter[4])
        # self.conv2 = nn.Conv2d(nb_filter[4], nb_filter[4], 3, padding=1)
        # self.bn2 = nn.BatchNorm2d(nb_filter[4])
        self.up4 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.up3 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.up2 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.up1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.outconv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):

        x0 = self.conv0_0(x)
        x1 = self.conv1_0(self.maxpool(x0))
        x2 = self.conv2_0(x1)
        x3 = self.conv3_0(x2)
        x4 = self.conv4_0(x3)
        # x, y1 = self.down1(x)
        # x, y2 = self.down2(x)
        # x, y3 = self.down3(x)
        # x, y4 = self.down4(x)
        # x = F.dropout2d(F.relu(self.bn1(self.conv1(x))))
        # x = F.dropout2d(F.relu(self.bn2(self.conv2(x))))
        x = self.up4(torch.cat([x3, self.up(x4)], 1))
        x = self.up3(torch.cat([x2, self.up(x)], 1))
        x = self.up2(torch.cat([x1, self.up(x)], 1))
        x = self.up1(torch.cat([x0, self.up(x)], 1))
        x1 = self.outconv(self.up(x))

        return x1