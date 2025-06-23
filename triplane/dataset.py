import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def make_coord(shape, ranges=None, flatten=True):
    """
    Make coordinates at grid centers.
    Here code is from https://github.com/yinboc/liif/blob/main/utils.py
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class IntensityDataset(Dataset):
    def __init__(self, dataset_path, points_batch_size,rand_size=True):
        super(IntensityDataset, self).__init__()
        self.dataset_path=dataset_path
        self.filenames = sorted(os.listdir(dataset_path))
        self.points_batch_size = points_batch_size
        self.rand_size=rand_size
        self.obj_data_ls = []
        for file in self.filenames:
            obj_data = np.load(os.path.join(self.dataset_path, file))
            self.obj_data_ls.append(obj_data)


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, obj_idx):
        obj_data = self.obj_data_ls[obj_idx]

        with torch.no_grad():
            obj_data = torch.from_numpy(obj_data / obj_data.max())#.to('cuda')
            size = obj_data.shape[0]
            if self.rand_size:
                min_size,max_size = int(size//2),int(size*2)
                size_z = np.random.randint(min_size, max_size)
                size_y = np.random.randint(min_size, max_size)
                size_x = np.random.randint(min_size, max_size)
                import torch.nn.functional as F
                obj_data = F.interpolate(obj_data.unsqueeze(0).unsqueeze(0), size=[size_z, size_y, size_x],mode='trilinear').squeeze()

            xyz = make_coord(obj_data.shape, flatten=True).flip(-1)#.to('cuda')
            sample_indices =np.random.choice(xyz.shape[0],self.points_batch_size)
            # sample_indices = torch.randperm(len(xyz),device='cpu')[:int(self.points_batch_size)] # faster but need more GPU memory
            xyz_sample = xyz[sample_indices]
            obj_data_sample = obj_data.reshape(-1, 1)[sample_indices]

        return obj_idx,xyz_sample,obj_data_sample