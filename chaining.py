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

def get_initial_traj(
        pred_flow, 
        pred_flow_rev, 
        query_t, 
        query_vis, 
        points_to_update,
    ):
    """
    Args:
        pred_flow: B, T, 2, H, W, forward flow from t to t + self.sliding_window_size - 1
        pred_flow_rev: B, T, 2, H, W, reverse flow from t + 1 to t + self.sliding_window_size
        query_t: B, N, T
        query_points: B, N, T, 2
        query_vis: B, N, T
        points_to_update: B, N
    Returns:
        p: B, N, T, 2
        vis: B, N, T
    """    
    B, T, H, W,_ = pred_flow.shape
    T=T+1
    cycle_threshold = 0.01 * min(H, W)
    
    pred_flow = pred_flow.permute(0,1,4,2,3)
    pred_flow_rev = pred_flow_rev.permute(0,1,4,2,3)

    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grids = torch.cat((xx, yy), 1).float().permute(0,2,3,1).reshape(B, -1, 2).cuda() # (B, N, 2)

    query_points = torch.cat((xx, yy), 1).float().permute(0,2,3,1).reshape(B, -1, 2).cuda() # (B, N, 2)
    
    WH = torch.tensor([W, H], device=query_points.device).float()

    query_points_norm = query_points / WH * 2 - 1 # (B, N, 2)
    p = repeat(query_points, 'b n c -> b n t c', t=T).clone()
    p_norm = repeat(query_points_norm, 'b n c -> b n t c', t=T).clone()
    vis = torch.ones_like(p[..., 0], dtype=torch.bool, device=query_points.device) # (B, N)

    for t, rev_t in zip(range(1, T), range(T - 2, -1, -1)):
        # update_forward = torch.logical_and(query_t < t, points_to_update) # (B, N)
        # update_backward = torch.logical_and(query_t > rev_t, points_to_update) # (B, N)

        # predict forward flow from t - 1 to t and reverse flow from t to t - 1
        pred_flow_p = F.grid_sample(
            pred_flow[:, t-1],
            rearrange(p_norm[:, :, t-1], 'b n c -> b n () c'),
            mode='bicubic', align_corners=False
        )
        pred_flow_p = rearrange(pred_flow_p, 'b c n () -> b n c')

        # pred_flow_p_rev = F.grid_sample(
        #     pred_flow_rev[:, rev_t],
        #     rearrange(p_norm[:, :, rev_t+1], 'b n c -> b n () c'),
        #     mode='bicubic', align_corners=False
        # )
        # pred_flow_p_rev = rearrange(pred_flow_p_rev, 'b c n () -> b n c')

        # update_forward_t = torch.logical_and(
        #     update_forward[:, :, None],
        #     (torch.arange(T, device=query_points.device) == t).reshape(1, 1, -1)
        # )
        # update_backward_t = torch.logical_and(
        #     update_backward[:, :, None],
        #     (torch.arange(T, device=query_points.device) == rev_t).reshape(1, 1, -1)
        # )
        
        p[:,:,t] = p[:, :, t-1] + pred_flow_p
        p_norm[:,:,t] = p_norm[:,:, t-1] + pred_flow_p / WH * 2

        # p[update_backward_t] = p[update_backward][:, rev_t+1] + pred_flow_p_rev[update_backward]
        # p_norm[update_backward_t] = p_norm[update_backward][:, rev_t+1] + pred_flow_p_rev[update_backward] / WH * 2

        # predict reverse flows for cycle consistency
        pred_flow_p_c = F.grid_sample(
            pred_flow_rev[:, t-1],
            rearrange(p_norm[:, :, t], 'b n c -> b n () c'),
            mode='bicubic', align_corners=False
        )
        pred_flow_p_c = rearrange(pred_flow_p_c, 'b c n () -> b n c')
        cycle_forward = torch.norm(pred_flow_p + pred_flow_p_c, dim=-1) < cycle_threshold # (B, N)

        # pred_flow_p_rev_c = F.grid_sample(
        #     pred_flow[:, rev_t],
        #     rearrange(p_norm[:, :, rev_t], 'b n c -> b n () c'),
        #     mode='bicubic', align_corners=False
        # )
        # pred_flow_p_rev_c = rearrange(pred_flow_p_rev_c, 'b c n () -> b n c')
        # cycle_backward = torch.norm(pred_flow_p_rev[update_backward] + pred_flow_p_rev_c[update_backward], dim=1) < cycle_threshold # (B, N)
        vis[:,:,t] = vis[:, :, t-1] & cycle_forward 
        # vis[update_backward_t] = vis[update_backward][:, rev_t+1] & cycle_backward
    grids = repeat(grids, 'b n c -> b n t c', t=T)
    p = p - grids
    p = p.reshape(B, H, W, T, 2).permute(0, 3, 1, 2, 4)
    vis = vis.reshape(B, H, W, T).permute(0, 3, 1, 2)

    return p, vis

@torch.no_grad()
def calc_flow_recursive_sintel(args, model, path, device=torch.device('cuda'), visualize=True):
    # Load images from the specified path
    scene_list = sorted(os.listdir(os.path.join(path,'flow')))
    for scene in scene_list:
        print('Processing', scene)
        
        rgb_folder_path = os.path.join(path, 'clean', scene)
        image_lists = sorted(os.listdir(rgb_folder_path))
        flow_folder_path = os.path.join(path, 'flow', scene)
        flow_lists = sorted(os.listdir(flow_folder_path))
        bw_flow_folder_path = os.path.join(path, 'bw_flow', scene)
        bw_flow_lists = sorted(os.listdir(bw_flow_folder_path))
        
        output_flow_path = os.path.join(path, 'chained_fw_flow', scene)
        os.makedirs(output_flow_path, exist_ok=True)
        output_occ_path = os.path.join(path, 'chained_fw_occ', scene)
        os.makedirs(output_occ_path, exist_ok=True)
        
        output_bw_flow_path = os.path.join(path, 'chained_bw_flow', scene)
        os.makedirs(output_bw_flow_path, exist_ok=True)
        output_bw_occ_path = os.path.join(path, 'chained_bw_occ', scene)
        os.makedirs(output_bw_occ_path, exist_ok=True)
        
        
        len_flow = len(flow_lists)
        
        ## forward flow
        for idx, flow_path in enumerate(flow_lists):
            end_idx = min(idx+9, len_flow)
            
            if idx>=end_idx:
                continue
            
            chaining_flow_list = flow_lists[idx:end_idx]
            chaining_flow_list = [os.path.join(flow_folder_path, flow) for flow in chaining_flow_list]
            flow_list = [read_gen(flow) for flow in chaining_flow_list]
            flow_list = [torch.tensor(flow).to(device) for flow in flow_list]
            flow_list = torch.stack(flow_list, dim=0).unsqueeze(dim=0)
            
            chaining_bw_flow_list = bw_flow_lists[idx:end_idx]
            chaining_bw_flow_list = [os.path.join(bw_flow_folder_path, flow) for flow in chaining_bw_flow_list]
            bw_flow_list = [read_gen(flow) for flow in chaining_bw_flow_list]
            bw_flow_list = [torch.tensor(flow).to(device) for flow in bw_flow_list]
            bw_flow_list = torch.stack(bw_flow_list, dim=0).unsqueeze(dim=0)
            
            chained_flows, vis = get_initial_traj(flow_list, bw_flow_list,None,None,None)
            
            if visualize:                
                image_list = image_lists[idx:end_idx+1]
                image_list = [os.path.join(rgb_folder_path, image) for image in image_list]
                image_list = [cv2.imread(image) for image in image_list]
                image_list = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in image_list]
                image_list = [torch.tensor(image).to(device) for image in image_list]
                image_list = torch.stack(image_list, dim=0).unsqueeze(dim=0).permute(0,1,4,2,3).float()
            
            for k in range(chained_flows.shape[1]):
                chained_flow = chained_flows[0,k]
                vis_vis = vis[0,k].unsqueeze(dim=0).repeat(3,1,1)
                
                if k!=0:
                    writeFlowKITTI(os.path.join(output_flow_path, f'{idx+1:04d}_{idx+k+1:04d}.png'), chained_flow.cpu().numpy())    
                    torchvision.utils.save_image(vis_vis.float(), os.path.join(output_occ_path, f'{idx:04d}_{idx+k:04d}.png'))
                    
                if visualize:
                    flow_vis = torchvision.utils.flow_to_image(chained_flow.permute(2,0,1))
                    flow_vis[~vis_vis]=0
                    
                    image1 = image_list[0,0]
                    image2 = image_list[0,k]
                    
                    warped_image = warp(image2.unsqueeze(dim=0), chained_flow.unsqueeze(dim=0).permute(0,3,1,2))
                    warped_image[~(vis_vis.unsqueeze(dim=0))]=0
                                        
                    torchvision.utils.save_image(flow_vis/255., f'./vis/chained_flow_{k}.png')
                    torchvision.utils.save_image(image1/255, f'./vis/image1.png')
                    torchvision.utils.save_image(image2/255, f'./vis/image{k}.png')
                    torchvision.utils.save_image(warped_image/255, f'./vis/warped_image_{k}.png')
            
        ## backward flow
        for idx in range(len_flow,-1,-1):
            start_idx = max(idx-9, 0)
            
            if idx<=start_idx:
                continue
            
            chaining_bw_flow_list = bw_flow_lists[start_idx:idx][::-1]
            chaining_bw_flow_list = [os.path.join(bw_flow_folder_path, flow) for flow in chaining_bw_flow_list]
            bw_flow_list = [read_gen(flow) for flow in chaining_bw_flow_list]
            bw_flow_list = [torch.tensor(flow).to(device) for flow in bw_flow_list]
            bw_flow_list = torch.stack(bw_flow_list, dim=0).unsqueeze(dim=0)
            
            chaining_fw_flow_list = flow_lists[start_idx:idx][::-1]
            chaining_fw_flow_list = [os.path.join(flow_folder_path, flow) for flow in chaining_fw_flow_list]
            fw_flow_list = [read_gen(flow) for flow in chaining_fw_flow_list]
            fw_flow_list = [torch.tensor(flow).to(device) for flow in fw_flow_list]
            fw_flow_list = torch.stack(fw_flow_list, dim=0).unsqueeze(dim=0)
            
            chained_flows, vis = get_initial_traj(bw_flow_list, fw_flow_list,None,None,None)
            
            if visualize:
                image_list = image_lists[start_idx:idx+1][::-1]
                image_list = [os.path.join(rgb_folder_path, image) for image in image_list]
                image_list = [cv2.imread(image) for image in image_list]
                image_list = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in image_list]
                image_list = [torch.tensor(image).to(device) for image in image_list]
                image_list = torch.stack(image_list, dim=0).unsqueeze(dim=0).permute(0,1,4,2,3).float()
                
            for k in range(chained_flows.shape[1]):
                chained_flow = chained_flows[0,k]
                vis_vis = vis[0,k].unsqueeze(dim=0).repeat(3,1,1)
                
                if k!=0:
                    writeFlowKITTI(os.path.join(output_bw_flow_path, f'{idx+1:04d}_{idx-k+1:04d}.png'), chained_flow.cpu().numpy())    
                    torchvision.utils.save_image(vis_vis.float(), os.path.join(output_bw_occ_path, f'{idx:04d}_{idx-k:04d}.png'))
                    
                
                if visualize:
                    flow_vis = torchvision.utils.flow_to_image(chained_flow.permute(2,0,1))
                    flow_vis[~vis_vis]=0
                    
                    image1 = image_list[0,0]
                    image2 = image_list[0,k]
                    
                    warped_image = warp(image2.unsqueeze(dim=0), chained_flow.unsqueeze(dim=0).permute(0,3,1,2))
                    warped_image[~(vis_vis.unsqueeze(dim=0))]=0
                                        
                    torchvision.utils.save_image(flow_vis/255., f'./vis/chained_flow_{k}.png')
                    torchvision.utils.save_image(image1/255, f'./vis/image1.png')
                    torchvision.utils.save_image(image2/255, f'./vis/image{k}.png')
                    torchvision.utils.save_image(warped_image/255, f'./vis/warped_image_{k}.png')
                    
@torch.no_grad()
def calc_flow_recursive_spring(args, model, path, device=torch.device('cuda'), visualize=False):
    # Load images from the specified path
    scene_list = sorted(os.listdir(path))
    output_path = '/mnt/data1/motion_dust3r_dataset/spring/train/'
    
    for scene in scene_list:
        print('Processing', scene)
        
        rgb_folder_path = os.path.join(path, scene, 'frame_left')
        image_lists = sorted(os.listdir(rgb_folder_path))
        flow_folder_path = os.path.join(path, scene, 'flow_FW_left')
        flow_lists = sorted(os.listdir(flow_folder_path))
        bw_flow_folder_path = os.path.join(path,scene,  'flow_BW_left')
        bw_flow_lists = sorted(os.listdir(bw_flow_folder_path))
        
        output_flow_path = os.path.join(output_path, scene, 'chained_fw_flow')
        os.makedirs(output_flow_path, exist_ok=True)
        output_occ_path = os.path.join(output_path,  scene, 'chained_fw_occ')
        os.makedirs(output_occ_path, exist_ok=True)
        
        output_bw_flow_path = os.path.join(output_path, scene , 'chained_bw_flow')
        os.makedirs(output_bw_flow_path, exist_ok=True)
        output_bw_occ_path = os.path.join(output_path, scene , 'chained_bw_occ')
        os.makedirs(output_bw_occ_path, exist_ok=True)
        
        len_flow = len(flow_lists)
        
        ## forward flow
        for idx, flow_path in enumerate(flow_lists):
            end_idx = min(idx+9, len_flow)
            
            if idx>=end_idx:
                continue
            
            chaining_flow_list = flow_lists[idx:end_idx]
            chaining_flow_list = [os.path.join(flow_folder_path, flow) for flow in chaining_flow_list]
            flow_list = [read_gen(flow) for flow in chaining_flow_list]
            flow_list = [torch.tensor(flow).to(device) for flow in flow_list]
            flow_list = torch.stack(flow_list, dim=0).unsqueeze(dim=0).float()
            
            flow_list = flow_list[:,:,::2,::2]
            ## nan to zero
            flow_list[torch.isnan(flow_list)] = 0
            
            chaining_bw_flow_list = bw_flow_lists[idx:end_idx]
            chaining_bw_flow_list = [os.path.join(bw_flow_folder_path, flow) for flow in chaining_bw_flow_list]
            bw_flow_list = [read_gen(flow) for flow in chaining_bw_flow_list]
            bw_flow_list = [torch.tensor(flow).to(device) for flow in bw_flow_list]
            bw_flow_list = torch.stack(bw_flow_list, dim=0).unsqueeze(dim=0).float()
            
            bw_flow_list = bw_flow_list[:,:,::2,::2]
            ## nan to zero
            bw_flow_list[torch.isnan(bw_flow_list)] = 0
            
            
            chained_flows, vis = get_initial_traj(flow_list, bw_flow_list,None,None,None)
            
            if visualize:                
                image_list = image_lists[idx:end_idx+1]
                image_list = [os.path.join(rgb_folder_path, image) for image in image_list]
                image_list = [cv2.imread(image) for image in image_list]
                image_list = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in image_list]
                image_list = [torch.tensor(image).to(device) for image in image_list]
                image_list = torch.stack(image_list, dim=0).unsqueeze(dim=0).permute(0,1,4,2,3).float()
            
            for k in range(chained_flows.shape[1]):
                chained_flow = chained_flows[0,k]
                vis_vis = vis[0,k].unsqueeze(dim=0).repeat(3,1,1)
                
                if k!=0:
                    writeFlowKITTI(os.path.join(output_flow_path, f'{idx+1:04d}_{idx+k+1:04d}.png'), chained_flow.cpu().numpy())    
                    torchvision.utils.save_image(vis_vis.float(), os.path.join(output_occ_path, f'{idx+1:04d}_{idx+k+1:04d}.png'))
                    
                if visualize:
                    flow_vis = torchvision.utils.flow_to_image(chained_flow.permute(2,0,1))
                    flow_vis[~vis_vis]=0
                    
                    image1 = image_list[0,0]
                    image2 = image_list[0,k]

                    warped_image = warp(image2.unsqueeze(dim=0), chained_flow.unsqueeze(dim=0).permute(0,3,1,2))
                    warped_image[~(vis_vis.unsqueeze(dim=0))]=0
                                        
                    torchvision.utils.save_image(flow_vis/255., f'./vis_spring/chained_flow_{k}.png')
                    torchvision.utils.save_image(image1/255, f'./vis_spring/image1.png')
                    torchvision.utils.save_image(image2/255, f'./vis_spring/image{k}.png')
                    torchvision.utils.save_image(warped_image/255, f'./vis_spring/warped_image_{k}.png')
            
        ## backward flow
        for idx in range(len_flow,-1,-1):
            start_idx = max(idx-9, 0)
            
            if idx<=start_idx:
                continue
            
            chaining_bw_flow_list = bw_flow_lists[start_idx:idx][::-1]
            chaining_bw_flow_list = [os.path.join(bw_flow_folder_path, flow) for flow in chaining_bw_flow_list]
            bw_flow_list = [read_gen(flow) for flow in chaining_bw_flow_list]
            bw_flow_list = [torch.tensor(flow).to(device) for flow in bw_flow_list]
            bw_flow_list = torch.stack(bw_flow_list, dim=0).unsqueeze(dim=0).float()
            
            bw_flow_list = bw_flow_list[:,:,::2,::2]
            ## nan to zero
            bw_flow_list[torch.isnan(bw_flow_list)] = 0
            
            chaining_fw_flow_list = flow_lists[start_idx:idx][::-1]
            chaining_fw_flow_list = [os.path.join(flow_folder_path, flow) for flow in chaining_fw_flow_list]
            fw_flow_list = [read_gen(flow) for flow in chaining_fw_flow_list]
            fw_flow_list = [torch.tensor(flow).to(device) for flow in fw_flow_list]
            fw_flow_list = torch.stack(fw_flow_list, dim=0).unsqueeze(dim=0).float()
            
            chained_flows, vis = get_initial_traj(bw_flow_list, fw_flow_list,None,None,None)
            
            fw_flow_list = fw_flow_list[:,:,::2,::2]
            ## nan to zero
            fw_flow_list[torch.isnan(fw_flow_list)] = 0
            
            if visualize:
                image_list = image_lists[start_idx:idx+1][::-1]
                image_list = [os.path.join(rgb_folder_path, image) for image in image_list]
                image_list = [cv2.imread(image) for image in image_list]
                image_list = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in image_list]
                image_list = [torch.tensor(image).to(device) for image in image_list]
                image_list = torch.stack(image_list, dim=0).unsqueeze(dim=0).permute(0,1,4,2,3).float()
                
            for k in range(chained_flows.shape[1]):
                chained_flow = chained_flows[0,k]
                vis_vis = vis[0,k].unsqueeze(dim=0).repeat(3,1,1)
                
                if k!=0:
                    writeFlowKITTI(os.path.join(output_bw_flow_path, f'{idx+1:04d}_{idx-k+1:04d}.png'), chained_flow.cpu().numpy())    
                    torchvision.utils.save_image(vis_vis.float(), os.path.join(output_bw_occ_path, f'{idx+1:04d}_{idx-k+1:04d}.png'))
                    
                
                if visualize:
                    flow_vis = torchvision.utils.flow_to_image(chained_flow.permute(2,0,1))
                    flow_vis[~vis_vis]=0
                    
                    image1 = image_list[0,0]
                    image2 = image_list[0,k]
                    
                    warped_image = warp(image2.unsqueeze(dim=0), chained_flow.unsqueeze(dim=0).permute(0,3,1,2))
                    warped_image[~(vis_vis.unsqueeze(dim=0))]=0
                                        
                    torchvision.utils.save_image(flow_vis/255., f'./vis_spring/chained_flow_{k}.png')
                    torchvision.utils.save_image(image1/255, f'./vis_spring/image1.png')
                    torchvision.utils.save_image(image2/255, f'./vis_spring/image{k}.png')
                    torchvision.utils.save_image(warped_image/255, f'./vis_spring/warped_image_{k}.png')
                
                
def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 100.0
    return flow, valid
                    
@torch.no_grad()
def calc_flow_recursive_point_odyssey(args, model, path, device=torch.device('cuda'), visualize=False, index=None):
    # Load images from the specified path
    scene_list = sorted(os.listdir(path))
    output_path = '/mnt/data1/motion_dust3r_dataset/point_odyssey/train/'
    
    if index is not None:
        scene_list = scene_list[index[0]:index[1]]
        
    for scene in scene_list:
        print('Processing', scene)
        
        rgb_folder_path = os.path.join(path, scene, 'rgbs')
        image_lists = sorted(os.listdir(rgb_folder_path))
        flow_folder_path = os.path.join(path, scene, 'fw_flow')
        flow_lists = sorted(os.listdir(flow_folder_path))
        bw_flow_folder_path = os.path.join(path,scene,  'bw_flow')
        bw_flow_lists = sorted(os.listdir(bw_flow_folder_path))
        
        output_flow_path = os.path.join(output_path, scene, 'chained_fw_flow')
        os.makedirs(output_flow_path, exist_ok=True)
        output_occ_path = os.path.join(output_path,  scene, 'chained_fw_occ')
        os.makedirs(output_occ_path, exist_ok=True)
        
        output_bw_flow_path = os.path.join(output_path, scene , 'chained_bw_flow')
        os.makedirs(output_bw_flow_path, exist_ok=True)
        output_bw_occ_path = os.path.join(output_path, scene , 'chained_bw_occ')
        os.makedirs(output_bw_occ_path, exist_ok=True)
        
        len_flow = len(flow_lists)
        
        ## forward flow
        for idx, flow_path in enumerate(flow_lists):
            end_idx = min(idx+9, len_flow)
            
            if idx>=end_idx:
                continue
            
            chaining_flow_list = flow_lists[idx:end_idx]
            chaining_flow_list = [os.path.join(flow_folder_path, flow) for flow in chaining_flow_list]
            flow_list = [readFlowKITTI(flow) for flow in chaining_flow_list]
            flow_list = [torch.tensor(flow[0]).to(device) for flow in flow_list]
            flow_list = torch.stack(flow_list, dim=0).unsqueeze(dim=0).float()
            
            flow_list[torch.isnan(flow_list)] = 0
            
            chaining_bw_flow_list = bw_flow_lists[idx:end_idx]
            chaining_bw_flow_list = [os.path.join(bw_flow_folder_path, flow) for flow in chaining_bw_flow_list]
            bw_flow_list = [readFlowKITTI(flow) for flow in chaining_bw_flow_list]
            bw_flow_list = [torch.tensor(flow[0]).to(device) for flow in bw_flow_list]
            bw_flow_list = torch.stack(bw_flow_list, dim=0).unsqueeze(dim=0).float()
            
            bw_flow_list[torch.isnan(bw_flow_list)] = 0
            
            
            chained_flows, vis = get_initial_traj(flow_list, bw_flow_list,None,None,None)
            
            if visualize:                
                image_list = image_lists[idx:end_idx+1]
                image_list = [os.path.join(rgb_folder_path, image) for image in image_list]
                image_list = [cv2.imread(image) for image in image_list]
                image_list = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in image_list]
                image_list = [torch.tensor(image).to(device) for image in image_list]
                image_list = torch.stack(image_list, dim=0).unsqueeze(dim=0).permute(0,1,4,2,3).float()
            
            for k in range(chained_flows.shape[1]):
                chained_flow = chained_flows[0,k]
                vis_vis = vis[0,k].unsqueeze(dim=0).repeat(3,1,1)
                
                if k!=0:
                    writeFlowKITTI(os.path.join(output_flow_path, f'{idx+1:04d}_{idx+k+1:04d}.png'), chained_flow.cpu().numpy())    
                    torchvision.utils.save_image(vis_vis.float(), os.path.join(output_occ_path, f'{idx+1:04d}_{idx+k+1:04d}.png'))
                    
                if visualize:
                    flow_vis = torchvision.utils.flow_to_image(chained_flow.permute(2,0,1))
                    flow_vis[~vis_vis]=0
                    
                    image1 = image_list[0,0]
                    image2 = image_list[0,k]

                    warped_image = warp(image2.unsqueeze(dim=0), chained_flow.unsqueeze(dim=0).permute(0,3,1,2))
                    warped_image[~(vis_vis.unsqueeze(dim=0))]=0
                                        
                    torchvision.utils.save_image(flow_vis/255., f'./vis_spring/chained_flow_{k}.png')
                    torchvision.utils.save_image(image1/255, f'./vis_spring/image1.png')
                    torchvision.utils.save_image(image2/255, f'./vis_spring/image{k}.png')
                    torchvision.utils.save_image(warped_image/255, f'./vis_spring/warped_image_{k}.png')
            
        ## backward flow
        for idx in range(len_flow,-1,-1):
            start_idx = max(idx-9, 0)
            
            if idx<=start_idx:
                continue
            
            chaining_bw_flow_list = bw_flow_lists[start_idx:idx][::-1]
            chaining_bw_flow_list = [os.path.join(bw_flow_folder_path, flow) for flow in chaining_bw_flow_list]
            bw_flow_list = [readFlowKITTI(flow) for flow in chaining_bw_flow_list]
            bw_flow_list = [torch.tensor(flow[0]).to(device) for flow in bw_flow_list]
            bw_flow_list = torch.stack(bw_flow_list, dim=0).unsqueeze(dim=0).float()
            
            chaining_fw_flow_list = flow_lists[start_idx:idx][::-1]
            chaining_fw_flow_list = [os.path.join(flow_folder_path, flow) for flow in chaining_fw_flow_list]
            fw_flow_list = [readFlowKITTI(flow) for flow in chaining_fw_flow_list]
            fw_flow_list = [torch.tensor(flow[0]).to(device) for flow in fw_flow_list]
            fw_flow_list = torch.stack(fw_flow_list, dim=0).unsqueeze(dim=0).float()
            
            chained_flows, vis = get_initial_traj(bw_flow_list, fw_flow_list,None,None,None)
            
            if visualize:
                image_list = image_lists[start_idx:idx+1][::-1]
                image_list = [os.path.join(rgb_folder_path, image) for image in image_list]
                image_list = [cv2.imread(image) for image in image_list]
                image_list = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in image_list]
                image_list = [torch.tensor(image).to(device) for image in image_list]
                image_list = torch.stack(image_list, dim=0).unsqueeze(dim=0).permute(0,1,4,2,3).float()
                
            for k in range(chained_flows.shape[1]):
                chained_flow = chained_flows[0,k]
                vis_vis = vis[0,k].unsqueeze(dim=0).repeat(3,1,1)
                
                if k!=0:
                    writeFlowKITTI(os.path.join(output_bw_flow_path, f'{idx+1:04d}_{idx-k+1:04d}.png'), chained_flow.cpu().numpy())    
                    torchvision.utils.save_image(vis_vis.float(), os.path.join(output_bw_occ_path, f'{idx+1:04d}_{idx-k+1:04d}.png'))
                    
                
                if visualize:
                    flow_vis = torchvision.utils.flow_to_image(chained_flow.permute(2,0,1))
                    flow_vis[~vis_vis]=0
                    
                    image1 = image_list[0,0]
                    image2 = image_list[0,k]
                    
                    warped_image = warp(image2.unsqueeze(dim=0), chained_flow.unsqueeze(dim=0).permute(0,3,1,2))
                    warped_image[~(vis_vis.unsqueeze(dim=0))]=0
                                        
                    torchvision.utils.save_image(flow_vis/255., f'./vis_spring/chained_flow_{k}.png')
                    torchvision.utils.save_image(image1/255, f'./vis_spring/image1.png')
                    torchvision.utils.save_image(image2/255, f'./vis_spring/image{k}.png')
                    torchvision.utils.save_image(warped_image/255, f'./vis_spring/warped_image_{k}.png')
 
from PIL import Image                   
def flowreader(flow_path):
    with Image.open(flow_path) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        flow = np.frombuffer(
            np.array(depth_pil, dtype=np.uint16), dtype=np.float16
        ).astype(np.float32).reshape((depth_pil.size[1], depth_pil.size[0]))
    flow_res = np.stack([flow[:,:flow.shape[1]//2], flow[:,flow.shape[1]//2:]],axis=-1)
    return flow_res
                    
@torch.no_grad()
def calc_flow_recursive_dynamic_replica(args, model, path, device=torch.device('cuda'), visualize=False):
    # Load images from the specified path
    scene_list = sorted(os.listdir(path))
    output_path = '/mnt/data1/motion_dust3r_dataset/dynamic_replica/'
    
    for scene in scene_list:
        print('Processing', scene)
        
        rgb_folder_path = os.path.join(path, scene, 'images')
        image_lists = sorted(os.listdir(rgb_folder_path))
        flow_folder_path = os.path.join(path, scene, 'flow_forward')
        flow_lists = sorted(os.listdir(flow_folder_path))
        bw_flow_folder_path = os.path.join(path,scene,  'flow_backward')
        bw_flow_lists = sorted(os.listdir(bw_flow_folder_path))
        
        output_flow_path = os.path.join(output_path, scene, 'chained_fw_flow')
        os.makedirs(output_flow_path, exist_ok=True)
        output_occ_path = os.path.join(output_path,  scene, 'chained_fw_occ')
        os.makedirs(output_occ_path, exist_ok=True)
        
        output_bw_flow_path = os.path.join(output_path, scene , 'chained_bw_flow')
        os.makedirs(output_bw_flow_path, exist_ok=True)
        output_bw_occ_path = os.path.join(output_path, scene , 'chained_bw_occ')
        os.makedirs(output_bw_occ_path, exist_ok=True)
        
        len_flow = len(flow_lists)
        
        ## forward flow
        for idx, flow_path in enumerate(flow_lists):
            end_idx = min(idx+9, len_flow)
            
            if idx>=end_idx:
                continue
            
            chaining_flow_list = flow_lists[idx:end_idx]
            chaining_flow_list = [os.path.join(flow_folder_path, flow) for flow in chaining_flow_list]
            flow_list = [flowreader(flow) for flow in chaining_flow_list]
            flow_list = [torch.tensor(flow).to(device) for flow in flow_list]
            flow_list = torch.stack(flow_list, dim=0).unsqueeze(dim=0).float()
            
            flow_list[torch.isnan(flow_list)] = 0
            
            chaining_bw_flow_list = bw_flow_lists[idx:end_idx]
            chaining_bw_flow_list = [os.path.join(bw_flow_folder_path, flow) for flow in chaining_bw_flow_list]
            bw_flow_list = [flowreader(flow) for flow in chaining_bw_flow_list]
            bw_flow_list = [torch.tensor(flow).to(device) for flow in bw_flow_list]
            bw_flow_list = torch.stack(bw_flow_list, dim=0).unsqueeze(dim=0).float()
            
            bw_flow_list[torch.isnan(bw_flow_list)] = 0
            
            
            chained_flows, vis = get_initial_traj(flow_list, bw_flow_list,None,None,None)
            
            if visualize:                
                image_list = image_lists[idx:end_idx+1]
                image_list = [os.path.join(rgb_folder_path, image) for image in image_list]
                image_list = [cv2.imread(image) for image in image_list]
                image_list = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in image_list]
                image_list = [torch.tensor(image).to(device) for image in image_list]
                image_list = torch.stack(image_list, dim=0).unsqueeze(dim=0).permute(0,1,4,2,3).float()
            
            for k in range(chained_flows.shape[1]):
                chained_flow = chained_flows[0,k]
                vis_vis = vis[0,k].unsqueeze(dim=0).repeat(3,1,1)
                
                if k!=0:
                    writeFlowKITTI(os.path.join(output_flow_path, f'{idx+1:04d}_{idx+k+1:04d}.png'), chained_flow.cpu().numpy())    
                    torchvision.utils.save_image(vis_vis.float(), os.path.join(output_occ_path, f'{idx+1:04d}_{idx+k+1:04d}.png'))
                    
                if visualize:
                    flow_vis = torchvision.utils.flow_to_image(chained_flow.permute(2,0,1))
                    flow_vis[~vis_vis]=0
                    
                    image1 = image_list[0,0]
                    image2 = image_list[0,k]

                    warped_image = warp(image2.unsqueeze(dim=0), chained_flow.unsqueeze(dim=0).permute(0,3,1,2))
                    warped_image[~(vis_vis.unsqueeze(dim=0))]=0
                                        
                    torchvision.utils.save_image(flow_vis/255., f'./vis_spring/chained_flow_{k}.png')
                    torchvision.utils.save_image(image1/255, f'./vis_spring/image1.png')
                    torchvision.utils.save_image(image2/255, f'./vis_spring/image{k}.png')
                    torchvision.utils.save_image(warped_image/255, f'./vis_spring/warped_image_{k}.png')
            
        ## backward flow
        for idx in range(len_flow,-1,-1):
            start_idx = max(idx-9, 0)
            
            if idx<=start_idx:
                continue
            
            chaining_bw_flow_list = bw_flow_lists[start_idx:idx][::-1]
            chaining_bw_flow_list = [os.path.join(bw_flow_folder_path, flow) for flow in chaining_bw_flow_list]
            bw_flow_list = [flowreader(flow) for flow in chaining_bw_flow_list]
            bw_flow_list = [torch.tensor(flow).to(device) for flow in bw_flow_list]
            bw_flow_list = torch.stack(bw_flow_list, dim=0).unsqueeze(dim=0).float()
            
            chaining_fw_flow_list = flow_lists[start_idx:idx][::-1]
            chaining_fw_flow_list = [os.path.join(flow_folder_path, flow) for flow in chaining_fw_flow_list]
            fw_flow_list = [flowreader(flow) for flow in chaining_fw_flow_list]
            fw_flow_list = [torch.tensor(flow).to(device) for flow in fw_flow_list]
            fw_flow_list = torch.stack(fw_flow_list, dim=0).unsqueeze(dim=0).float()
            
            chained_flows, vis = get_initial_traj(bw_flow_list, fw_flow_list,None,None,None)
            
            if visualize:
                image_list = image_lists[start_idx:idx+1][::-1]
                image_list = [os.path.join(rgb_folder_path, image) for image in image_list]
                image_list = [cv2.imread(image) for image in image_list]
                image_list = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in image_list]
                image_list = [torch.tensor(image).to(device) for image in image_list]
                image_list = torch.stack(image_list, dim=0).unsqueeze(dim=0).permute(0,1,4,2,3).float()
                
            for k in range(chained_flows.shape[1]):
                chained_flow = chained_flows[0,k]
                vis_vis = vis[0,k].unsqueeze(dim=0).repeat(3,1,1)
                
                if k!=0:
                    writeFlowKITTI(os.path.join(output_bw_flow_path, f'{idx+1:04d}_{idx-k+1:04d}.png'), chained_flow.cpu().numpy())    
                    torchvision.utils.save_image(vis_vis.float(), os.path.join(output_bw_occ_path, f'{idx+1:04d}_{idx-k+1:04d}.png'))
                    
                
                if visualize:
                    flow_vis = torchvision.utils.flow_to_image(chained_flow.permute(2,0,1))
                    flow_vis[~vis_vis]=0
                    
                    image1 = image_list[0,0]
                    image2 = image_list[0,k]
                    
                    warped_image = warp(image2.unsqueeze(dim=0), chained_flow.unsqueeze(dim=0).permute(0,3,1,2))
                    warped_image[~(vis_vis.unsqueeze(dim=0))]=0
                                        
                    torchvision.utils.save_image(flow_vis/255., f'./vis_spring/chained_flow_{k}.png')
                    torchvision.utils.save_image(image1/255, f'./vis_spring/image1.png')
                    torchvision.utils.save_image(image2/255, f'./vis_spring/image{k}.png')
                    torchvision.utils.save_image(warped_image/255, f'./vis_spring/warped_image_{k}.png')
            


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--device', help='inference device', type=str, default='cuda')
    parser.add_argument('--dataset', help='dataset', type=str, default='sintel')
    parser.add_argument('--indexes', type=int, nargs='+', default=[0,131])
    args = parser.parse_args()
    
    
    if args.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    if args.dataset == 'sintel':
        calc_flow_recursive_sintel(args, None, '/mnt/data4/motion_dust3r_dataset/sintel/training/', device=device)
    elif args.dataset == 'spring':
        calc_flow_recursive_spring(args, None, '/mnt/data4/motion_dust3r_dataset/spring/train', device=device)
    elif args.dataset=='point_odyssey':
        calc_flow_recursive_point_odyssey(args, None, '/mnt/data4/motion_dust3r_dataset/point_odyssey/train', device=device, index=args.indexes)
    elif args.dataset=='dynamic_replica':
        calc_flow_recursive_dynamic_replica(args, None, '/mnt/data3/motion_dust3r_dataset/dynamic_replica', device=device)

if __name__ == '__main__':
    main()