#!/usr/bin/python

"""
Train a diffusion model on images.
"""
import os
import argparse
from guided_diffusion import dist_util,logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from tensorboardX import SummaryWriter
import torch


def main():
    args = create_argparser().parse_args()
    writer = SummaryWriter(os.path.join(args.tblogger_path, args.exp_name))
    logger.configure(args.logger_path + "/" + args.exp_name)

    # Log hyperparameter configuration
    print(f'hparam dict: {args_to_dict(args, model_and_diffusion_defaults().keys())}')

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model = model.to(dist_util.dev(None))
    print(f'Number of devices: {torch.cuda.device_count()}')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    print("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        explicit_normalization=args.explicit_normalization,
        stats_dir=args.stats_dir,
    )

    print("training...")
    print(f'ckpt file will saved in {args.checkpoint_path+"/"+args.exp_name}')
    print(f'Visual sample file will saved in {args.visual_path+"/"+args.exp_name}')
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        writer=writer,
        explicit_normalization=args.explicit_normalization,
        stats_dir=args.stats_dir,
        ckpt_path=args.checkpoint_path + "/" + args.exp_name,
        val_path=args.visual_path + "/" + args.exp_name,
    ).run_loop()


def create_argparser():
    defaults = dict(
        exp_name="diffusion_training",
        # path
        data_dir="path/to/triplane_embeddings",
        checkpoint_path='path/to/ckpts',
        visual_path='path/to/visual',
        tblogger_path="path/to/tblogger",
        logger_path="path/to/logger",
        stats_dir=None,
        # train
        resume_checkpoint=None,
        schedule_sampler="uniform",
        explicit_normalization=False,
        lr=1e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        # interval
        log_interval=10,
        val_interval=5000,
        save_interval=5000,
        # fp
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )

    os.makedirs(os.path.join(defaults['tblogger_path'],defaults['exp_name']), exist_ok=True)
    os.makedirs(os.path.join(defaults['checkpoint_path'],defaults['exp_name']), exist_ok=True)
    os.makedirs(os.path.join(defaults['visual_path'],defaults['exp_name']), exist_ok=True)

    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    main()