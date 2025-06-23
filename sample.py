# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

import os

import torch
import argparse
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from skimage import io
import tifffile
import torchvision.transforms as transforms
from PIL import Image
import random
import time

from diffusion.guided_diffusion import dist_util, logger
from diffusion.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from zeroshot.condition_methods import get_conditioning_method
from triplane.networks import *
from triplane.dataset import make_coord
from zeroshot.physics.ct import *
from zeroshot.physics.mri import *


def main():
    args = create_argparser().parse_args()
    logger.configure(args.logger_path + "/" + args.exp_name)
    device = ('cuda' if th.cuda.is_available() else 'cpu')

    # load diffusion model
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev(None))
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # load decoder model
    logger.log("creating decoder model...")
    decoder = TriplaneConvAutoDecoder(resolution=128, channels=32, how_many_scenes=9, aggregate_fn='concat', use_tanh=True)

    decoder.load_state_dict(th.load(args.decoder_path)['model_state_dict'], strict=False)
    decoder.to(dist_util.dev(None))
    decoder.eval()

    # Prepare Operator and noise (same as zeroshot/clahe.py)
    def get_operator(deg_type):
        if deg_type == 'CT_LA':
            deg_op = CT_LA(radon_angle=90).A
        elif deg_type == 'CT_SV':
            deg_op = CT_SV(radon_view=36).A
        elif deg_type == 'MRI_CS':
            deg_op = MRI_CS(acc_factor=8).A
        elif deg_type == 'MRI_ZAxis':
            deg_op = MRI_ZAxis(zsr_factor=4).A
        return deg_op

    operator = get_operator(args.deg)
    noiser = 'gaussian'
    cond_method = get_conditioning_method('ps', operator, noiser, decoder,**{'scale':args.cond_scale,'sample_num':args.sample_num})

    # measurement guided sampling
    logger.log("sampling...")
    data_ls = sorted(os.listdir(args.y_path))
    for idx in range(0,len(data_ls)):
        # load data
        data=tifffile.imread(os.path.join(args.y_path,data_ls[idx]))
        y=torch.from_numpy(data).unsqueeze(1).to(torch.float32).to(dist_util.dev(None))

        # guidance settings
        model_kwargs = {}
        model_kwargs['deg'] = args.deg
        model_kwargs['cond_scale'] = args.cond_scale
        model_kwargs['measurement'] = y
        model_kwargs['measurement_cond_fn'] = cond_method.conditioning
        model_kwargs['savedir_y'] = args.visual_path+"/"+args.exp_name+'_'+str(args.cond_scale)+'/y'
        model_kwargs['savedir_ADx'] = args.visual_path+"/"+args.exp_name+'_'+str(args.cond_scale)+'/ADx'

        # diffusion sampling
        tic = time.time()
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        x_start = torch.randn((args.batch_size, args.triplane_size[-1]*3, args.image_size, args.image_size), device=device).requires_grad_()
        triplane = sample_fn(
            model,
            (args.batch_size, args.triplane_size[-1]*3, args.image_size, args.image_size),
            noise=x_start,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )
        tac = time.time()
        print("Sampling Triplane costs{:.2f} min".format((tac - tic) / 60))

        # decoder sampling
        with torch.no_grad():
            tic = time.time()
            sample_shape=args.sample_size
            xyz = make_coord(sample_shape, flatten=True).flip(-1).unsqueeze(0).to(device)
            gen = np.zeros((xyz.shape[1], 1))
            coord_ls = range(0, xyz.shape[1], args.coord_bs)
            for k in tqdm(coord_ls, desc='Coord Batch'):
                start = k
                end = min(k + args.coord_bs, xyz.shape[1])
                xyz_batch = xyz[:, start:end, :]
                gen_batch = decoder.decoder(triplane, xyz_batch)
                gen[start:end] = gen_batch.cpu().detach().numpy()
            sample = gen.reshape(sample_shape)

        # normalization
        print(sample.min(), sample.max())
        sample[sample>1]=1
        print(sample.min(),sample.max())

        # save images
        save_dir = args.visual_path+"/"+args.exp_name+'_'+str(args.cond_scale)
        os.makedirs(save_dir, exist_ok=True)
        io.imsave(os.path.join(save_dir, args.deg+ "%03d"%(idx+1) + '.tif'), (sample * 255).astype('uint8'))
        tac = time.time()
        print("Decode to Image costs{:.2f} min".format((tac - tic) / 60))

        # calculate metrics
        if args.gt_path:
            from triplane.metric import calculate_metrics
            image=io.imread(os.path.join(args.gt_path,data_ls[idx]))
            assert(image.shape==sample.shape)
            image=image/image.max()
            calculate_metrics(sample, image, save_dir=save_dir, vis_error=True, is_cuda=False)

        # logger
        logger.log(f"created {idx} samples")
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        exp_name="sample_demo",
        # paths
        model_path='path/to/diffusion_model.pt',
        decoder_path="path/to/triplane_decoder.pt",
        visual_path='path/to/visual',
        logger_path="path/to/logger",
        gt_path="path/to/gt_data_dir",
        y_path="path/to/degraded_data_dir",
        # diffusion model
        triplane_size=(128,128,96),
        batch_size=1,
        clip_denoised=True,
        use_ddim=False,
        # decoder
        coord_bs=500000,
        sample_size=(256,256,256),
        # guidance-based sampling
        deg='CT_SV',
        cond_scale=6,
        sample_num=16,
    )

    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
