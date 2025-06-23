import os

import tifffile
import torch
import numpy as np
import math
import json
from skimage import io
import torch.nn as nn
from pytorch_msssim import ssim,ms_ssim

def compute_psnr(arr1, arr2,need_2d=True):
    psnr_ls=[]
    mse_3d=nn.MSELoss()(arr1, arr2)
    psnr_3d = 20 * math.log10(1 / math.sqrt(mse_3d.item()))
    psnr_ls.append(psnr_3d)
    if need_2d:
        psnr_xy,psnr_xz,psnr_yz=0,0,0
        for i in range(arr1.shape[2]):
            mse_xy = nn.MSELoss()(arr1[:,:,i, :, :], arr2[:,:,i, :, :])
            psnr_xy +=  20 * math.log10(1 / (math.sqrt(mse_xy.item())+1e-6))
        for j in range(arr1.shape[3]):
            mse_xz = nn.MSELoss()(arr1[:, :, :, j, :], arr2[:, :, :, j, :])
            psnr_xz += 20 * math.log10(1 / (math.sqrt(mse_xz.item())+1e-6))
        for k in range(arr1.shape[4]):
            mse_yz = nn.MSELoss()(arr1[:, :, : ,:, k], arr2[:, :, :, :, k])
            psnr_yz += 20 * math.log10(1 / (math.sqrt(mse_yz.item())+1e-6))
        psnr_xy,psnr_xz,psnr_yz =psnr_xy/arr1.shape[2],psnr_xz / arr1.shape[3],psnr_yz / arr1.shape[4]
        psnr_ls.extend([psnr_xy,psnr_xz,psnr_yz])

    return psnr_ls

def compute_ssim(arr1, arr2,need_2d=True):
    ssim_ls=[]
    ssim_3d = ssim(arr1, arr2,win_size=11,data_range=1)
    ssim_ls.append(ssim_3d.item())
    if need_2d:
        ssim_xy,ssim_xz,ssim_yz=0,0,0
        for i in range(arr1.shape[2]):
            ssim_xy += ssim(arr1[:,:,i, :, :], arr2[:,:,i, :, :],win_size=11,data_range=1)
        for j in range(arr1.shape[3]):
            ssim_xz += ssim(arr1[:,:,:, j, :], arr2[:,:,:, j, :],win_size=11,data_range=1)
        for k in range(arr1.shape[4]):
            ssim_yz += ssim(arr1[:,:,:, :, k], arr2[:,:,:, :, k],win_size=11,data_range=1)
        ssim_xy,ssim_xz,ssim_yz =ssim_xy/arr1.shape[2],ssim_xz / arr1.shape[3],ssim_yz / arr1.shape[4]
        ssim_ls.extend([ssim_xy.item(),ssim_xz.item(),ssim_yz.item()])

    return ssim_ls

def compute_ms_ssim(arr1, arr2,need_2d=True):
    ms_ssim_ls=[]
    ms_ssim_3d = ms_ssim(arr1, arr2,win_size=5, data_range=1)
    ms_ssim_ls.append(ms_ssim_3d.item())
    if need_2d:
        ms_ssim_xy,ms_ssim_xz,ms_ssim_yz=0,0,0
        for i in range(arr1.shape[2]):
            ms_ssim_xy += ms_ssim(arr1[:,:,i, :, :], arr2[:,:,i, :, :],win_size=5, data_range=1)
        for j in range(arr1.shape[3]):
            ms_ssim_xz += ms_ssim(arr1[:,:,:, j, :], arr2[:,:,:, j, :],win_size=5, data_range=1)
        for k in range(arr1.shape[4]):
            ms_ssim_yz += ms_ssim(arr1[:,:,:, :, k], arr2[:,:,:, :, k],win_size=5, data_range=1)
        ms_ssim_xy,ms_ssim_xz,ms_ssim_yz =ms_ssim_xy/arr1.shape[2],ms_ssim_xz / arr1.shape[3],ms_ssim_yz / arr1.shape[4]
        ms_ssim_ls.extend([ms_ssim_xy.item(),ms_ssim_xz.item(),ms_ssim_yz.item()])

    return ms_ssim_ls


def calculate_metrics(arr1,arr2,save_dir,name='metric.json',vis_error=True,is_cuda=False):
    assert arr1.shape == arr2.shape
    arr1 = torch.tensor(arr1[np.newaxis, np.newaxis, ...])
    arr2 = torch.tensor(arr2[np.newaxis, np.newaxis, ...])
    if is_cuda:
        arr1=arr1.cuda()
        arr2=arr2.cuda()

    metrics={}
    metrics['psnr'] = compute_psnr(arr1, arr2,need_2d=True)
    print('psnr:',metrics['psnr'])
    metrics['ssim']=compute_ssim(arr1, arr2,need_2d=True)
    print('ssim:',metrics['ssim'])

    if vis_error:
        err = torch.abs(arr1 - arr2)
        err_np = np.array(err.squeeze() * 255).astype('uint8')
        io.imsave(os.path.join(save_dir, 'error.tif'), err_np)

    save_json=os.path.join(save_dir,name)
    with open(save_json, "w") as f:
        f.write(json.dumps(metrics, ensure_ascii=False, indent=4, separators=(',', ':')))

    return metrics