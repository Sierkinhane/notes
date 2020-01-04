""" Full assembly of the parts to form the complete network """
import cv2
import torch.nn.functional as F
import torch.nn as nn
from .unet_parts import *
import matplotlib.pyplot as plt

def data_visulization(score_map):
    score_map = score_map.data.cpu().numpy()
    bs, nc = score_map.shape[:2]
    for i in range(0,  nc, 1):
        m = score_map[0, i, :,:]
        # print(m.shape)
        print(i)
        # m = cv2.resize(m, (0,0), fx=256/64, fy=256/64, interpolation=cv2.INTER_CUBIC)
        colormap = cv2.applyColorMap(m, cv2.COLORMAP_HSV)
        plt.axis('off')
        # plt.imshow(img)
        plt.imshow(colormap, alpha=0.5)
        # plt.plot(landmarks_coords[i,0], landmarks_coords[i,1], 'r.', ms=7)
        # plt.savefig('../../Heatmap-finetune-speed-up-network/heatmaps/pred_colormap_' + str(i) + '.jpg')
        plt.show()
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)         # (64, 128)
        self.down2 = Down(128, 256)       # (128,128)
        self.down3 = Down(256, 512)       # (128,256)
        self.down4 = Down(512, 512)       # (256, 256)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        # self.ouc = nn.Conv2d(64, 1, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNet_small(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_small, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)         # (64, 128)
        self.down2 = Down(128, 128)       # (128,128)
        self.down3 = Down(128, 256)       # (128,256)
        self.down4 = Down(256, 256)       # (256, 256)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)  #256+
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class Stacked_UNet(nn.Module):
    """docstring for Stacked_UNet"""
    def __init__(self):
        super(Stacked_UNet, self).__init__()
        self.u1 = UNet_uultras(1,1, cat=True)
        self.u2 = UNet_uultras(18,1)

    def forward(self, x):
        x1, x_con = self.u1(x)
        x2 = self.u2(torch.cat([x, x1, x_con], 1))

        return x1, x2
        

def build_stacked_unet(num):
    model = nn.Sequential(
        UNet_small(1,1),
        UNet_small()
        )

class UNet_ultras(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_ultras, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)         # (32, 64)
        self.down2 = Down(64, 64)       # (64,64)
        self.down3 = Down(64, 128)       # (128,128)
        self.down4 = Down(128, 128)       # (128, 128)
        self.up1 = Up(256, 64, bilinear)
        self.up2 = Up(128, 64, bilinear)
        self.up3 = Up(128, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)  #256+
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.sigmoid(self.outc(x))
        # data_visulization(logits)
        # data_visulization(x)
        return logits

class UNet_uultras(nn.Module):
    def __init__(self, n_channels, n_classes, cat=False, bilinear=True):
        super(UNet_uultras, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.cat = cat
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)         # (32, 32)
        self.down2 = Down(32, 32)       # (32,32)
        self.down3 = Down(32, 64)       # (64,64)
        self.down4 = Down(64, 64)       # (64, 64)
        self.up1 = Up(128, 32, bilinear)
        self.up2 = Up(64, 32, bilinear)
        self.up3 = Up(64, 16, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)  #256+
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.sigmoid(self.outc(x))
        # data_visulization(logits)
        if self.cat:
            return logits, x
        else:
            return logits

class UNet_Attention_ultras(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_Attention_ultras, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)         # (32, 64)
        self.down2 = Down(64, 64)       # (64,64)
        self.down3 = Down(64, 128)       # (128,128)
        self.down4 = Down(128, 128)       # (128, 128)
        self.up1 = Up(256, 64, bilinear)
        self.up2 = Up(128, 64, bilinear)
        self.up3 = Up(128, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.attentionblock1 = GridAttentionBlock2D(in_channels=32, inter_channels=16, gating_channels=32, mode='concatenation', sub_sample_factor=(2,2))
        self.attentionblock2 = GridAttentionBlock2D(in_channels=64, inter_channels=16, gating_channels=64, mode='concatenation', sub_sample_factor=(2,2))
        self.attentionblock3 = GridAttentionBlock2D(in_channels=64, inter_channels=16, gating_channels=64, mode='concatenation', sub_sample_factor=(2,2))
        self.outc = OutConv(32, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)

        g_conv3, att3 = self.attentionblock3(x3, x)
        x = self.up2(x, g_conv3)  #256+
        g_conv2, att2 = self.attentionblock2(x2, x)
        x = self.up3(x, g_conv2)
        g_conv1, att1 = self.attentionblock1(x1, x)
        x = self.up4(x, g_conv1)
        logits = self.sigmoid(self.outc(x))
        return logits

def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))

if __name__ == '__main__':
    import numpy as np 
    import torch
    # from utils import model_info
    # model = UNet_small(1, 1) # 4.10e+06
    # model = UNet_small(1, 1)               # 1.34e+07
    model = Stacked_UNet()
    # model = UNet_uultras(1, 1)
    model_info(model)
    inp = torch.from_numpy(np.random.normal(0, 1, [68, 1, 64, 64]).astype(np.float32))
    preds, _ = model(inp)
    print(_.shape)