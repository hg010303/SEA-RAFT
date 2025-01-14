import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat
import argparse
import os
import cv2
import numpy as np
import torchvision
from core.utils.frame_utils import read_gen, writeFlowKITTI

def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 100.0
    return flow, valid

def warp(x, flo, padding_mode='zeros', return_mask=False):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    args:
        x: [B, C, H, W]
        flo: [B, 2, H, W] flow
    outputs:
        output: warped x [B, C, H, W]
    """
    B, C, H, W = flo.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # makes a mapping out of the flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)

    output = nn.functional.grid_sample(x, vgrid, align_corners=True, padding_mode=padding_mode)

    if return_mask:
        vgrid = vgrid.permute(0, 3, 1, 2)
        mask = (vgrid[:, 0] > -1) & (vgrid[:, 1] > -1) & (vgrid[:, 0] < 1) & (vgrid[:, 1] < 1)
        return output, mask
    return output

path = '/mnt/data4/motion_dust3r_dataset/sintel/training/'
scene = 'alley_1'

rgb_folder_path = os.path.join(path, 'clean', scene)
image_lists = sorted(os.listdir(rgb_folder_path))
chained_flow_folder_path = os.path.join(path, 'chained_fw_flow', scene)
chained_bw_folder_path = os.path.join(path, 'chained_bw_flow', scene)
chained_occ_folder_path = os.path.join(path, 'chained_fw_occ', scene)
chained_bw_occ_folder_path = os.path.join(path, 'chained_bw_occ', scene)


image_idx1 = 5
image_idx2 = 14

image1 = cv2.imread(os.path.join(rgb_folder_path, image_lists[image_idx1]))
image2 = cv2.imread(os.path.join(rgb_folder_path, image_lists[image_idx2]))

chained_flow,_ = readFlowKITTI(os.path.join(chained_flow_folder_path, f'{image_idx1+1:04d}_{image_idx2+1:04d}.png'))
chained_bw,_ = readFlowKITTI(os.path.join(chained_bw_folder_path, f'{image_idx2+1:04d}_{image_idx1+1:04d}.png'))

chained_occ = cv2.imread(os.path.join(chained_occ_folder_path, f'{image_idx1+1:04d}_{image_idx2+1:04d}.png'), cv2.IMREAD_GRAYSCALE)
chained_bw_occ = cv2.imread(os.path.join(chained_bw_occ_folder_path, f'{image_idx2+1:04d}_{image_idx1+1:04d}.png'), cv2.IMREAD_GRAYSCALE)

image1 = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
image2 = torch.from_numpy(image2).permute(2, 0, 1).unsqueeze(0).float() / 255.0

chained_flow = torch.from_numpy(chained_flow).permute(2, 0, 1).unsqueeze(0)
chained_bw = torch.from_numpy(chained_bw).permute(2, 0, 1).unsqueeze(0)

chained_occ = torch.from_numpy(chained_occ).unsqueeze(0).unsqueeze(0).float() / 255.0
chained_bw_occ = torch.from_numpy(chained_bw_occ).unsqueeze(0).unsqueeze(0).float() / 255.0

fw_warped = warp(image2, chained_flow)
bw_warped = warp(image1, chained_bw)

fw_warped = fw_warped * (chained_occ)
bw_warped = bw_warped * (chained_bw_occ)

torchvision.utils.save_image(fw_warped, 'fw_warped.png')
torchvision.utils.save_image(bw_warped, 'bw_warped.png')
torchvision.utils.save_image(image1, 'image1.png')
torchvision.utils.save_image(image2, 'image2.png')