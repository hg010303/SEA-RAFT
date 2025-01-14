import numpy as np
from PIL import Image
from os.path import *
import re
import h5py
import cv2
import torch
import torchvision

def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 100.0
    return flow, valid

path = "/mnt/data3/motion_dust3r_dataset/point_odyssey/train/ani11_new_/fw_flow/fw_flow_00001.png"

flow, valid = readFlowKITTI(path)

flow_fw_vis = torchvision.utils.flow_to_image(torch.tensor(flow).unsqueeze(dim=0).permute(0,3, 1, 2))
torchvision.utils.save_image(flow_fw_vis/255., f'./flow_fw.png')
