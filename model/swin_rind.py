import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .swin_transformer import SwinTransformer

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        # create backbone
        # backbone_cfg =dict(
        #     embed_dim=192,
        #     depths=[2, 2, 18, 2],
        #     num_heads=[6, 12, 24, 48],
        #     window_size=12,
        #     drop_path_rate=0.3
        # )
        # swin v1
        backbone_cfg = dict(
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
        )
        self.backbone = SwinTransformer(**backbone_cfg)

        # bimal decoder
        self.deconv_t2 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.deconv_t3 = nn.ConvTranspose2d(768, 192, kernel_size=4, stride=4)
        self.deconv_t4 = nn.ConvTranspose2d(1536, 192, kernel_size=8, stride=8)

        # top-down
        self.top_down_1 = nn.Conv2d(192, 1, kernel_size=3, padding=1)
        self.top_down_2 = nn.Conv2d(192, 1, kernel_size=3, padding=1)
        self.top_down_3 = nn.Conv2d(192, 1, kernel_size=3, padding=1)
        self.top_down_4 = nn.Conv2d(192, 1, kernel_size=3, padding=1)

        self.top_down_1_2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=4)
        self.top_down_2_2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=4)
        self.top_down_3_2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=4)
        self.top_down_4_2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=4)
        # bottom-up
        self.bottom_up_1 = nn.Conv2d(192, 1, kernel_size=3, padding=1)
        self.bottom_up_2 = nn.Conv2d(192, 1, kernel_size=3, padding=1)
        self.bottom_up_3 = nn.Conv2d(192, 1, kernel_size=3, padding=1)
        self.bottom_up_4 = nn.Conv2d(192, 1, kernel_size=3, padding=1)

        self.bottom_up_1_2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=4)
        self.bottom_up_2_2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=4)
        self.bottom_up_3_2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=4)
        self.bottom_up_4_2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=4)

        # conv stack for all edges
        self.conv_stack_reflectance = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.conv_stack_illumination = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.conv_stack_normal = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.conv_stack_depth = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        # decision head for all edges
        self.DH_reflectance = nn.Conv2d(8, 1, kernel_size=1)
        self.DH_illumination = nn.Conv2d(8, 1, kernel_size=1)
        self.DH_normal = nn.Conv2d(8, 1, kernel_size=1)
        self.DH_depth = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        feats = self.backbone(x)

        # BiMAL decoder
        t1 = feats[0]
        t2 = self.deconv_t2(feats[1])
        t3 = self.deconv_t3(feats[2])
        t4 = self.deconv_t4(feats[3])

        # top-down
        top_down_1 = self.top_down_1(t4)
        top_down_2 = self.top_down_2(t4 + t3)
        top_down_3 = self.top_down_3(t4 + t3 + t2)
        top_down_4 = self.top_down_4(t4 + t3 + t2 + t1)

        f_1 = self.top_down_1_2(top_down_1)
        f_2 = self.top_down_2_2(top_down_2)
        f_3 = self.top_down_3_2(top_down_3)
        f_4 = self.top_down_4_2(top_down_4)

        # bottom-up
        bottom_up_1 = self.bottom_up_1(t1 + t2 + t3 + t4)
        bottom_up_2 = self.bottom_up_2(t1 + t2 + t3)
        bottom_up_3 = self.bottom_up_3(t1 + t2)
        bottom_up_4 = self.bottom_up_4(t1)

        f_5 = self.bottom_up_1_2(bottom_up_1)
        f_6 = self.bottom_up_2_2(bottom_up_2)
        f_7 = self.bottom_up_3_2(bottom_up_3)
        f_8 = self.bottom_up_4_2(bottom_up_4)

        concat = torch.cat([f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8], dim=1)

        feature_reflectance = self.conv_stack_reflectance(concat)
        feature_illumination = self.conv_stack_illumination(concat)
        feature_normal = self.conv_stack_normal(concat)
        feature_depth = self.conv_stack_depth(concat)
        #
        # # decision head
        DH_reflectance = self.DH_reflectance(feature_reflectance)
        DH_illumination = self.DH_illumination(feature_illumination)
        DH_normal = self.DH_normal(feature_normal)
        DH_depth = self.DH_depth(feature_depth)

        # sigmoid edges
        out_reflectance = torch.sigmoid(DH_reflectance)
        out_illumination = torch.sigmoid(DH_illumination)
        out_normal = torch.sigmoid(DH_normal)
        out_depth = torch.sigmoid(DH_depth)

        viz = [out_depth, out_normal, out_reflectance, out_illumination]

        return out_depth, out_normal, out_reflectance, out_illumination, viz



# pre_test
if __name__ == '__main__':
    backbone_cfg = dict(
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        drop_path_rate=0.3
    )
    model = SwinTransformer(**backbone_cfg)
    checkpoint = torch.load(r'C:\Users\cvl\Desktop\miao\RINDNet-main\model_zoo\swinv2_large_patch4_window12_192_22k.pth')
    state_dict = model.state_dict()
    state_dict.update(checkpoint['model'])
    model.load_state_dict(state_dict)
    dummy_input = torch.rand(1, 3, 256, 256)
    output = model(dummy_input)
    print(output)