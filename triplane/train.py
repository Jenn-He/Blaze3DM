import os
import argparse
import numpy as np
import torch
from dataset import *
from torch.utils.data import DataLoader
from networks import *
from tensorboardX import SummaryWriter
from tqdm import tqdm
import tifffile
import math
from loss import *

import gc
import torch.cuda.amp.autocast_mode
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='triplane fitting')
    parser.add_argument('--exp_name', type=str, default="autodecoder_demo", help='name of experiment')
    # paths
    parser.add_argument('--data_dir', type=str, default="path/to/volume_data_dir",
                    help='directory to high-quality medical volumes for training')
    parser.add_argument('--val_data', type=str, default="path/to/validation_data.tif",
                    help='directory to validation volume')
    parser.add_argument('--checkpoint_path', type=str, default='path/to/ckpts',
                    help='where to save model checkpoints')
    parser.add_argument('--visual_path', type=str, default='path/to/visual',
                    help='where to save triplane visualizations')
    parser.add_argument('--tblogger_path', type=str, default='path/to/tblogger',
                    help='where to save tensorboard logger')
    # triplane
    parser.add_argument('--resolution', type=int, default=128, help='resolution size of triplane')
    parser.add_argument('--channels', type=int, default=32, help='channel size of triplane')
    parser.add_argument('--aggregate_fn', type=str, default='concat',
                    help='function for aggregating triplane features')
    parser.add_argument('--use_tanh', default=True, help='Whether to use tanh to clamp triplanes to [-1, 1].')
    # train
    parser.add_argument('--load_ckpt_path', type=str, default=None, help='checkpoint to continue training from')
    parser.add_argument('--batch_size', type=int, default=1, help='number of batch per training step')
    parser.add_argument('--points_batch_size', type=int, default=1000000, help='number of points per volume')
    parser.add_argument('--steps_per_batch', type=int, default=10, help='If specified, how many GD steps to run on a batch before moving on. To address I/O stuff.')
    parser.add_argument('--rand_size', type=bool, default=True, help='random size augmentation')
    parser.add_argument('--lambda_tv', type=float, default=1e-2, help='weight for TV regularization')
    parser.add_argument('--lambda_l2', type=float, default=1e-3, help='weight for L2 regularization')
    parser.add_argument('--lambda_eir', type=float, default=1, help='weight for eir regularization')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate of training process')
    parser.add_argument('--epochs', type=int, default=500000, help='Training epoch, A big number -- can easily do early stopping with Ctrl+C.')
    # interval
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--vis_every', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=50)

    args = parser.parse_args()
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    os.makedirs(os.path.join(args.tblogger_path,args.exp_name), exist_ok=True)
    os.makedirs(os.path.join(args.checkpoint_path,args.exp_name), exist_ok=True)
    os.makedirs(os.path.join(args.visual_path,args.exp_name), exist_ok=True)
    writer = SummaryWriter(os.path.join(args.tblogger_path, args.exp_name))

    # Load the entire dataset onto GPU
    train_dataloader = DataLoader(
        IntensityDataset(dataset_path=args.data_dir, points_batch_size=args.points_batch_size, rand_size=args.rand_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Triplane auto-decoder
    how_many_scenes = len(train_dataloader)
    auto_decoder = TriplaneConvAutoDecoder(resolution=args.resolution, channels=args.channels, how_many_scenes=how_many_scenes,
                                       aggregate_fn=args.aggregate_fn, use_tanh=args.use_tanh)

    # train triplane and mlp together
    optimizer = torch.optim.Adam(params=auto_decoder.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    auto_decoder.train()
    ckpt_epoch=0
    loss_fn = lambda x,y: torch.nn.MSELoss()(x,y) # torch.nn.MSELoss()(x,y)+torch.nn.L1Loss()(x,y)

    if args.load_ckpt_path:
        ckpt_epoch = int(args.load_ckpt_path.split('_')[-1].split('.')[0])
        checkpoint = torch.load(args.load_ckpt_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        auto_decoder.load_state_dict(checkpoint['model_state_dict'], strict=False)

    N_EPOCHS = args.epochs
    step = 0
    scaler = GradScaler()
    for epoch in range(ckpt_epoch+1,N_EPOCHS+1):
        print(f'EPOCH {epoch}...')
        for iter,batch in enumerate(train_dataloader, 1):
            obj_idx, coordinates, gt_intensity = batch[0],batch[1].type(Tensor),batch[2].type(Tensor)
            print(obj_idx)

            for _step in range(args.steps_per_batch):
                optimizer.zero_grad()
                with autocast():
                    pred_intensity = auto_decoder(obj_idx, coordinates)
                    pixel_loss = loss_fn(pred_intensity, gt_intensity)

                    tv_reg = auto_decoder.tvreg()
                    l2_reg = auto_decoder.l2reg()
                    eir_reg = eir_loss(obj_idx, auto_decoder)

                    loss =  pixel_loss + tv_reg* args.lambda_tv+l2_reg*args.lambda_l2+eir_reg*args.lambda_eir

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step += 1

        gc.collect()
        # torch.cuda.empty_cache()

        if not epoch % args.log_every:
            print(f'Epoch {epoch}: Loss {loss.item()}')
            writer.add_scalar('loss', loss.item(), epoch)
            writer.add_scalar('pixel_loss', pixel_loss.item(), epoch)
            writer.add_scalar('tv_reg', tv_reg.item(), epoch)
            writer.add_scalar('l2_reg', l2_reg.item(), epoch)
            writer.add_scalar('eir_reg', eir_reg.item(), epoch)

        if not epoch % args.save_every:
            print(f'Saving checkpoint at epoch {epoch}')
            torch.save({
                'model_state_dict': auto_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },f'{args.checkpoint_path}/{args.exp_name}/model_epoch_{epoch}.pt')

        if not epoch % args.vis_every:
            with torch.no_grad():
                auto_decoder.eval()

                image = np.load(args.val_data)
                image = image / image.max()
                xyz = make_coord(image.shape, flatten=True).flip(-1).unsqueeze(0).to(device)
                idx = torch.from_numpy(np.array([0])).to(device)
                gen = np.zeros((xyz.shape[1], 1))
                # coordinate batch
                coord_ls = range(0, xyz.shape[1], args.points_batch_size)
                for k in tqdm(coord_ls, desc='Coord Batch'):
                    start = k
                    end = min(k + args.points_batch_size, xyz.shape[1])
                    xyz_batch = xyz[:, start:end, :]
                    gen_batch = auto_decoder(idx, xyz_batch)
                    gen[start:end] = gen_batch.cpu().detach().numpy()
                gen = gen.reshape(image.shape)
                metric = np.sum((gen-image)**2)/len(gen.reshape(-1))
                metric = 20 * math.log10(1 / math.sqrt(metric))
                print(f'Epoch {epoch}: Metric {metric}')
                writer.add_scalar('metric', metric, epoch)
                tifffile.imsave(os.path.join(args.visual_path, args.exp_name, '%04d'% epoch + '.tif'), (gen / gen.max() * 255).astype('uint8'))

                gc.collect()
                # torch.cuda.empty_cache()

                auto_decoder.train()

