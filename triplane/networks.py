import torch
import torch.nn as nn
import numpy as np
import math


class SinActivation(nn.Module):
    def forward(self, x):
        return (torch.sin(x) + 1) / 2

class PosEncoding(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength=None,  use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            #assert fn_samples is not None
            fn_samples = sidelength
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)
        else:
            self.num_frequencies = 4

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)


class TriplaneConv(nn.Module):
    def __init__(self, channels, out_channels, kernel_size, padding, is_rollout=True) -> None:
        super().__init__()
        in_channels = channels * 3 if is_rollout else channels
        self.is_rollout = is_rollout

        self.conv_xy = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_yz = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_xz = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, featmaps):
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_yz, tpl_xz = featmaps[:,0],featmaps[:,1],featmaps[:,2]
        H, W = tpl_xy.shape[-2:]
        D = tpl_xz.shape[-1]

        tpl_xy_h = torch.cat([tpl_xy,
                           torch.mean(tpl_yz, dim=-1, keepdim=True).transpose(-1, -2).expand_as(tpl_xy),
                           torch.mean(tpl_xz, dim=-1, keepdim=True).expand_as(tpl_xy)], dim=1)  # [B, C * 3, H, W]
        tpl_yz_h = torch.cat([tpl_yz,
                           torch.mean(tpl_xy, dim=-2, keepdim=True).transpose(-1, -2).expand_as(tpl_yz),
                           torch.mean(tpl_xz, dim=-2, keepdim=True).expand_as(tpl_yz)], dim=1)  # [B, C * 3, W, D]
        tpl_xz_h = torch.cat([tpl_xz,
                           torch.mean(tpl_xy, dim=-1, keepdim=True).expand_as(tpl_xz),
                           torch.mean(tpl_yz, dim=-2, keepdim=True).expand_as(tpl_xz)], dim=1)  # [B, C * 3, H, D]

        assert tpl_xy_h.shape[-2] == H and tpl_xy_h.shape[-1] == W
        assert tpl_yz_h.shape[-2] == W and tpl_yz_h.shape[-1] == D
        assert tpl_xz_h.shape[-2] == H and tpl_xz_h.shape[-1] == D

        tpl_xy_h = self.conv_xy(tpl_xy_h)
        tpl_yz_h = self.conv_yz(tpl_yz_h)
        tpl_xz_h = self.conv_xz(tpl_xz_h)

        return torch.stack([tpl_xy_h,tpl_yz_h,tpl_xz_h], dim=1)


class TriplaneConvAutoDecoder(nn.Module):
    def __init__(self,resolution,channels,how_many_scenes,aggregate_fn='prod',use_tanh=False):
        super().__init__()

        self.aggregate_fn = aggregate_fn
        print(f'Using aggregate_fn {aggregate_fn}')

        self.resolution = resolution
        self.channels = channels
        self.embeddings = nn.ParameterList([torch.nn.Embedding(1, 3 * self.channels * self.resolution * self.resolution) for i in
                           range(how_many_scenes)])
        self.use_tanh = use_tanh
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        for embedding in self.embeddings:
            embedding.to(self.device)
        self.positional_encoding = PosEncoding(in_features=3,sidelength=512)

        self.conv= TriplaneConv(channels, channels, 3, padding=1, is_rollout=True).to(self.device)

        self.net = nn.Sequential(
            nn.Linear(self.channels*3+self.positional_encoding.out_dim, 128),
            SinActivation(),

            nn.Linear(128, 128),
            SinActivation(),

            nn.Linear(128, 1),
            SinActivation(),
        ).to(self.device)

    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords2d.reshape(coords2d.shape[0], 1, -1,
                                                                            coords2d.shape[-1]),
                                                           mode='bilinear', align_corners=False)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H * W).permute(0, 2, 1)
        return sampled_features

    def forward(self, obj_idx, coordinates):
        batch_size, n_coords, n_dims = coordinates.shape
        assert batch_size == obj_idx.shape[0]

        # Get embedding at index and reshape to (N, 3, channels, H, W)
        triplanes=self.embeddings[obj_idx.to('cpu')](torch.tensor(0, dtype=torch.int64).to(self.device)).view(batch_size, 3, self.channels, self.resolution, self.resolution)

        import tifffile
        import os
        tifffile.imwrite(os.path.join("PATH/TO/SAVE/triplane.tif"),triplanes.view(3*self.channels, self.resolution, self.resolution).detach().cpu().numpy())

        # 3d-aware triplane conv
        triplanes = self.conv(triplanes)+triplanes

        # Use tanh to clamp triplanes
        if self.use_tanh:
            triplanes = torch.tanh(triplanes)

        # Make sure all these coordinates line up.
        xy_embed = self.sample_plane(coordinates[..., 0:2], triplanes[:, 0])
        yz_embed = self.sample_plane(coordinates[..., 1:3], triplanes[:, 1])
        xz_embed = self.sample_plane(coordinates[..., :3:2], triplanes[:, 2])

        # Triplane aggregating fn.
        if self.aggregate_fn == 'sum':
            features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
        elif self.aggregate_fn == 'prod':
            features = torch.prod(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
        else: # 'concat'
            features = torch.concat([xy_embed, yz_embed, xz_embed], dim=-1)

        # decoder
        return self.net(torch.concat([self.positional_encoding(coordinates),features],dim=-1))

    def decoder(self,triplanes,coordinates):
        batch_size, n_coords, n_dims = coordinates.shape
        triplanes=triplanes.view(batch_size, 3, self.channels, self.resolution, self.resolution)
        triplanes = self.conv(triplanes)+triplanes

        if self.use_tanh:
            triplanes = torch.tanh(triplanes)

        # Make sure all these coordinates line up.
        xy_embed = self.sample_plane(coordinates[..., 0:2], triplanes[:, 0])
        yz_embed = self.sample_plane(coordinates[..., 1:3], triplanes[:, 1])
        xz_embed = self.sample_plane(coordinates[..., :3:2], triplanes[:, 2])

        # Triplane aggregating fn.
        if self.aggregate_fn == 'sum':
            features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
        elif self.aggregate_fn == 'prod':
            features = torch.prod(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
        else: # 'concat'
            features = torch.concat([xy_embed, yz_embed, xz_embed], dim=-1)

        # decoder
        return self.net(torch.concat([self.positional_encoding(coordinates),features],dim=-1))


    def tvreg(self):
        l = 0
        for embed in self.embeddings:
            triplane = embed(torch.tensor(0, dtype=torch.int64).to(self.device)).view(3, self.channels, self.resolution, self.resolution)
            l += ((triplane[:, :, 1:, :] - triplane[:, :, :-1, :]) ** 2).mean() ** 0.5
            l += ((triplane[:, :, :, 1:] - triplane[:, :, :, :-1]) ** 2).mean() ** 0.5
        return l / len(self.embeddings)

    def l2reg(self):
        l = 0
        for embed in self.embeddings:
            triplane = embed(torch.tensor(0, dtype=torch.int64).to(self.device)).view(3, self.channels, self.resolution,self.resolution)
            l += (triplane ** 2).mean() ** 0.5
        return l / len(self.embeddings)