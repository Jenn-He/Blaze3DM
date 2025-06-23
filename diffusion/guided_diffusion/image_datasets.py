import math
import random

from PIL import Image
import blobfile as bf
# from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch as th
import os


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    deterministic=False,
    random_rotate=False,
    random_flip=False,
    explicit_normalization=False,
    stats_dir=None,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    print(f'loading from: {data_dir}')
    print(f'number of files: {len(all_files)}')
    dataset = ImageDataset(
        image_size,
        all_files,
        normalize=explicit_normalization,
        random_rotate=random_rotate,
        random_flip=random_flip,
        stats_dir=stats_dir,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npy","pt"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        normalize=False,  # Whether to rescale individual channels to [-1, 1] based on their respective ranges
        shard=0,
        num_shards=1,
        random_rotate=False,
        random_flip=False,
        mix_up=False,
        stats_dir=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.normalize = normalize
        self.random_rotate = random_rotate
        self.random_flip = random_flip
        self.stats_dir = stats_dir
        self.mix_up=mix_up

        if self.normalize:
            print('Will normalize triplanes in the training loop.')
            if self.stats_dir is None:
                raise Exception('Need to provide a directory of stats to use for normalization.')
            # Load in min and max numpy arrays (shape==[96,] - one value per channel) for normalization
            # self.min_values = np.load(os.path.join(self.stats_dir, 'min_values.npy')).astype(np.float32).reshape(-1, 1, 1) # should be (96, 1, 1)
            # self.max_values = np.load(os.path.join(self.stats_dir, 'max_values.npy')).astype(np.float32).reshape(-1, 1, 1)
            self.min_values = np.load(os.path.join(self.stats_dir,'lower_bound.npy')).astype(np.float32).reshape(-1, 1, 1)
            self.max_values = np.load(os.path.join(self.stats_dir,'upper_bound.npy')).astype(np.float32).reshape(-1, 1, 1)
            self.range = self.max_values - self.min_values
            self.middle = (self.min_values + self.max_values) / 2
        else:
            print('Not using normalization in ds.')

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]

        # Load np array
        arr = np.load(path)
        arr = arr.astype(np.float32)
        arr = arr.reshape([-1, arr.shape[-2], arr.shape[-1]])

        # Normalize individual channels
        # / 127.5 - 1  <-- need to normalize the triplanes in their own way.
        if self.normalize:
            arr = (arr - self.middle) / (self.range / 2)

        out_dict = {}
        return arr, out_dict

    def unnormalize(self, sample):
        sample = sample * (self.range / 2) + self.middle
        return sample


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
