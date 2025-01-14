import sys
sys.path.append('core')
import argparse
import os
import cv2
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision

from config.parser import parse_args

import datasets
from raft import RAFT
from utils.flow_viz import flow_to_image
from utils.utils import load_ckpt
from utils.frame_utils import writeFlowKITTI, writeFlo5File

from tqdm.auto import tqdm

TAG_CHAR = np.array([202021.25], np.float32)
def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()
    
def readFlow(fn):
        """ Read .flo file in Middlebury format"""
        # Code adapted from:
        # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

        # WARNING: this will work on little-endian architectures (eg Intel x86) only!
        # print 'fn = %s'%(fn)
        with open(fn, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print('Magic number incorrect. Invalid .flo file')
                return None
            else:
                w = np.fromfile(f, np.int32, count=1)
                h = np.fromfile(f, np.int32, count=1)
                # print 'Reading %d x %d flo file\n' % (w, h)
                data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
                # Reshape data into 3D array (columns, rows, bands)
                # The reshape here is for visualization, the original code is (w,h,2)
                return np.resize(data, (int(h), int(w), 2))

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


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img
    
def check_cycle_consistency(flow_01, flow_10):
    # flow_01 = torch.from_numpy(flow_01).permute(2, 0, 1)[None]
    # flow_10 = torch.from_numpy(flow_10).permute(2, 0, 1)[None]
    H, W = flow_01.shape[-2:]
    coords = coords_grid(1, H, W, flow_01.device)
    coords1 = coords + flow_01
    flow_reprojected = bilinear_sampler(flow_10, coords1.permute(0, 2, 3, 1))
    cycle = flow_reprojected + flow_01
    cycle = torch.norm(cycle, dim=1)
    mask = (cycle < 0.01 * min(H, W)).float()
    return mask[0].detach().cpu().numpy()

def create_color_bar(height, width, color_map):
    """
    Create a color bar image using a specified color map.

    :param height: The height of the color bar.
    :param width: The width of the color bar.
    :param color_map: The OpenCV colormap to use.
    :return: A color bar image.
    """
    # Generate a linear gradient
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.repeat(gradient[np.newaxis, :], height, axis=0)

    # Apply the colormap
    color_bar = cv2.applyColorMap(gradient, color_map)

    return color_bar

def add_color_bar_to_image(image, color_bar, orientation='vertical'):
    """
    Add a color bar to an image.

    :param image: The original image.
    :param color_bar: The color bar to add.
    :param orientation: 'vertical' or 'horizontal'.
    :return: Combined image with the color bar.
    """
    if orientation == 'vertical':
        return cv2.vconcat([image, color_bar])
    else:
        return cv2.hconcat([image, color_bar])

def vis_heatmap(name, image, heatmap):
    # theta = 0.01
    # print(heatmap.max(), heatmap.min(), heatmap.mean())
    heatmap = heatmap[:, :, 0]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    # heatmap = heatmap > 0.01
    heatmap = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = image * 0.3 + colored_heatmap * 0.7
    # Create a color bar
    height, width = image.shape[:2]
    color_bar = create_color_bar(50, width, cv2.COLORMAP_JET)  # Adjust the height and colormap as needed
    # Add the color bar to the image
    overlay = overlay.astype(np.uint8)
    combined_image = add_color_bar_to_image(overlay, color_bar, 'vertical')
    cv2.imwrite(name, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

def get_heatmap(info, args):
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)              
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
    heatmap = (log_b * weight).sum(dim=1, keepdim=True)
    return heatmap

def forward_flow(args, model, image1, image2):
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final

def calc_flow(args, model, image1, image2):
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    H, W = img1.shape[2:]
    flow, info = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    return flow_down, info_down

@torch.no_grad()
def demo_data(path, args, model, image1, image2):
    os.system(f"mkdir -p {path}")
    H, W = image1.shape[2:]
    flow, info = calc_flow(args, model, image1, image2)
    flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    cv2.imwrite(f"{path}flow.jpg", flow_vis)
    heatmap = get_heatmap(info, args)
    vis_heatmap(f"{path}heatmap.jpg", image1[0].permute(1, 2, 0).cpu().numpy(), heatmap[0].permute(1, 2, 0).cpu().numpy())

@torch.no_grad()
def demo_custom(model, args, device=torch.device('cuda')):
    image1 = cv2.imread("./custom/image1.jpg")
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread("./custom/image2.jpg")
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
    image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
    H, W = image1.shape[1:]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    demo_data('./custom/', args, model, image1, image2)

@torch.no_grad()
def calc_flow_recursive(args, model, path, device=torch.device('cuda')):
    # Load images from the specified path
    scene_list = sorted(os.listdir(path))
    scene_list = [os.path.join(path, image) for image in scene_list]
    # import ipdb;ipdb.set_trace()
    for scene in scene_list:
        # if scene.split('/')[-1] != 'ani9_new_':
        #     continue
        # rgb_path =
        print('Processing', scene)
        rgb_folder_path = os.path.join(scene, 'rgbs')
        image_list = sorted(os.listdir(rgb_folder_path))
        
        for image1_path, image2_path in zip(image_list[:-9], image_list[9:]):
            image1 = cv2.imread(os.path.join(rgb_folder_path, image1_path))
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
            image1 = image1[None].to(device)
            
            image2 = cv2.imread(os.path.join(rgb_folder_path, image2_path))
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
            image2 = image2[None].to(device)
            import ipdb;ipdb.set_trace()
            fw_flow, info = calc_flow(args, model, image1, image2)
            bw_flow, info = calc_flow(args, model, image2, image1)
            
            os.makedirs(os.path.join(scene, 'fw_flow'), exist_ok=True)
            fw_flow_path = os.path.join(scene, 'fw_flow')
            
            # fw_flow_npy = fw_flow[0].cpu().numpy()
            # fw_flow_npy = fw_flow_npy.transpose(1, 2, 0)
            # fw_flow_path = os.path.join(fw_flow_path, image1_path.split('.')[0].replace('rgb','fw_flow') + '.png')
            # writeFlowKITTI(fw_flow_path, fw_flow_npy)



            # os.makedirs(os.path.join(scene, 'bw_flow'), exist_ok=True)
            # bw_flow_path = os.path.join(scene, 'bw_flow')
            
            # bw_flow_npy = bw_flow[0].cpu().numpy()
            # bw_flow_npy = bw_flow_npy.transpose(1, 2, 0)
            # bw_flow_path = os.path.join(bw_flow_path, image2_path.split('.')[0].replace('rgb','bw_flow') + '.png')
            # writeFlowKITTI(bw_flow_path, bw_flow_npy)
            
            torchvision.utils.save_image(image1, f'./image1.png', normalize=True)
            torchvision.utils.save_image(image2, f'./image2.png', normalize=True)
            # 
            bw_flow_vis = torchvision.utils.flow_to_image(bw_flow)
            torchvision.utils.save_image(bw_flow_vis/255., f'./flow_bw.png')
            fw_flow_vis = torchvision.utils.flow_to_image(fw_flow)
            torchvision.utils.save_image(fw_flow_vis/255., f'./flow_fw.png')
            
            
            ## check consistency
            fw_mask = check_cycle_consistency(fw_flow, bw_flow)
            bw_mask = check_cycle_consistency(bw_flow, fw_flow)
            
            fw_mask = 1-fw_mask
            bw_mask = 1-bw_mask
            

            
            
            
            # os.makedirs(os.path.join(scene, 'consistency_fw_mask'), exist_ok=True)
            # mask_path = os.path.join(scene, 'consistency_fw_mask')
            # fw_mask_vis = torch.tensor(fw_mask).unsqueeze(dim=0).unsqueeze(dim=0)
            # fw_mask_vis = fw_mask_vis.repeat(1,3,1,1)
            # torchvision.utils.save_image(fw_mask_vis, os.path.join(mask_path, image1_path.split('.')[0] + '.png'))
            # # torchvision.utils.save_image(fw_mask_vis, f'./fw_mask.png')
            
            # os.makedirs(os.path.join(scene, 'consistency_bw_mask'), exist_ok=True)
            # mask_path = os.path.join(scene, 'consistency_bw_mask')
            # bw_mask_vis = torch.tensor(bw_mask).unsqueeze(dim=0).unsqueeze(dim=0)
            # bw_mask_vis = bw_mask_vis.repeat(1,3,1,1)
            # torchvision.utils.save_image(bw_mask_vis,os.path.join(mask_path, image2_path.split('.')[0] + '.png'))
            # # torchvision.utils.save_image(bw_mask_vis, f'./bw_mask.png')
            
            # bw_mask_vis = torch.tensor(bw_mask).unsqueeze(dim=0).unsqueeze(dim=0)
            # bw_mask_vis = bw_mask_vis.repeat(1,3,1,1)
            # torchvision.utils.save_image(bw_mask_vis, f'./bw_mask.png')
            


@torch.no_grad()
def calc_flow_recursive_sintel(args, model, path, device=torch.device('cuda')):
    # Load images from the specified path
    scene_list = sorted(os.listdir(os.path.join(path,'flow')))
    for scene in scene_list:
        print('Processing', scene)
        rgb_folder_path = os.path.join(path, 'clean', scene)
        image_list = sorted(os.listdir(rgb_folder_path))
        
        for image1_path, image2_path in zip(image_list[:-10], image_list[10:]):
            image1 = cv2.imread(os.path.join(rgb_folder_path, image1_path))
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
            image1 = image1[None].to(device)
            
            image2 = cv2.imread(os.path.join(rgb_folder_path, image2_path))
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
            image2 = image2[None].to(device)
            
            
            fw_flow, info = calc_flow(args, model, image1, image2)
            bw_flow, info = calc_flow(args, model, image2, image1)
            
            ## vis flows
            # bw_flow_vis = torchvision.utils.flow_to_image(bw_flow)
            # torchvision.utils.save_image(bw_flow_vis/255., f'./flow_fw.png')
            torchvision.utils.save_image(image1, f'./image1.png', normalize=True)
            torchvision.utils.save_image(image2, f'./image2.png', normalize=True)

            # Save flow into .npz file
            # os.makedirs(os.path.join(path, 'bw_flow', scene), exist_ok=True)
            # bw_flow_path = os.path.join(path, 'bw_flow', scene)
            
            ## save flow as .flo file
            # bw_flow = bw_flow[0].cpu().numpy()
            # # bw_flow = bw_flow.transpose(1, 2, 0)
            # # bw_flow_path = os.path.join(bw_flow_path, image2_path.split('.')[0] + '.flo')
            # writeFlow(bw_flow_path, bw_flow)
            
            
            ## load .flo file
            # fw_flow_path = os.path.join(path, 'flow', scene, image1_path.split('.')[0] + '.flo')
            # fw_flow = readFlow(fw_flow_path)
            
            bw_flow_vis = torchvision.utils.flow_to_image(bw_flow)
            torchvision.utils.save_image(bw_flow_vis/255., f'./flow_bw.png')
            fw_flow_vis = torchvision.utils.flow_to_image(fw_flow)
            torchvision.utils.save_image(fw_flow_vis/255., f'./flow_fw.png')
            

            
            
            ## check consistency
            fw_mask = check_cycle_consistency(fw_flow, bw_flow)
            bw_mask = check_cycle_consistency(bw_flow, fw_flow)
            
            fw_mask = 1-fw_mask
            bw_mask = 1-bw_mask
            
            
            
            fw_warped_image = warp(image2, fw_flow).detach().cpu()
            bw_warped_image = warp(image1, bw_flow).detach().cpu()
            
            fw_warped_image = fw_warped_image * torch.tensor(1-fw_mask).unsqueeze(dim=0).unsqueeze(dim=0)
            bw_warped_image = bw_warped_image * torch.tensor(1-bw_mask).unsqueeze(dim=0).unsqueeze(dim=0)
            
            torchvision.utils.save_image(fw_warped_image, f'./fw_warped_image.png', normalize=True)
            torchvision.utils.save_image(bw_warped_image, f'./bw_warped_image.png', normalize=True)
            
            import ipdb;ipdb.set_trace()
            
            # os.makedirs(os.path.join(path, 'consistency_fw_mask', scene), exist_ok=True)
            # mask_path = os.path.join(path, 'consistency_fw_mask', scene)
            # fw_mask_vis = torch.tensor(fw_mask).unsqueeze(dim=0).unsqueeze(dim=0)
            # fw_mask_vis = fw_mask_vis.repeat(1,3,1,1)
            # torchvision.utils.save_image(fw_mask_vis, os.path.join(mask_path, image1_path.split('.')[0] + '.png'))
            
            # os.makedirs(os.path.join(path, 'consistency_bw_mask', scene), exist_ok=True)
            # mask_path = os.path.join(path, 'consistency_bw_mask', scene)
            # bw_mask_vis = torch.tensor(bw_mask).unsqueeze(dim=0).unsqueeze(dim=0)
            # bw_mask_vis = bw_mask_vis.repeat(1,3,1,1)
            # torchvision.utils.save_image(bw_mask_vis,os.path.join(mask_path, image2_path.split('.')[0] + '.png'))
            
            # bw_mask_vis = torch.tensor(bw_mask).unsqueeze(dim=0).unsqueeze(dim=0)
            # bw_mask_vis = bw_mask_vis.repeat(1,3,1,1)
            # torchvision.utils.save_image(bw_mask_vis, f'./bw_mask.png')
            
            ## save mask
            # os.makedirs(os.path.join(path, 'consistency_fw_mask', scene), exist_ok=True)
            # mask_path = os.path.join(path, 'consistency_fw_mask', scene)
            # np.save(os.path.join(mask_path, image1_path.split('.')[0] + '.npy'), fw_mask)
            
            # os.makedirs(os.path.join(path, 'consistency_bw_mask', scene), exist_ok=True)
            # mask_path = os.path.join(path, 'consistency_bw_mask', scene)
            # np.save(os.path.join(mask_path, image2_path.split('.')[0] + '.npy'), bw_mask)
            
        pass
    
    image_list = [cv2.imread(image) for image in image_list]
    image_list = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in image_list]
    image_list = [torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) for image in image_list]
    H, W = image_list[0].shape[1:]

    for idx, image in enumerate(tqdm(image_list)):
        image = image[None].to(device)
        image1 = image
        image2 = image
        flow, info = calc_flow(args, model, image1, image2)
        # Save flow into .npz file 
        np.savez_compressed(f"{path}/flow_{idx:04d}.npz", flow=flow[0].cpu().numpy())
        # Save visualization
        flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
        cv2.imwrite(f"{path}/flow_{idx:04d}.jpg", flow_vis)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--path', help='checkpoint path', type=str, default=None)
    parser.add_argument('--url', help='checkpoint url', type=str, default=None)
    parser.add_argument('--device', help='inference device', type=str, default='cpu')
    parser.add_argument('--dataset', help='dataset', type=str, default='sintel')
    args = parse_args(parser)
    if args.path is None and args.url is None:
        raise ValueError("Either --path or --url must be provided")
    if args.path is not None:
        model = RAFT(args)
        load_ckpt(model, args.path)
    else:
        model = RAFT.from_pretrained(args.url, args=args)
        
    if args.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    if args.dataset == 'sintel':
        calc_flow_recursive_sintel(args, model, '/mnt/data3/motion_dust3r_dataset/sintel/training')
    else:
        calc_flow_recursive(args, model, '/mnt/data3/motion_dust3r_dataset/point_odyssey/train')
    # demo_custom(model, args, device=device)

if __name__ == '__main__':
    main()