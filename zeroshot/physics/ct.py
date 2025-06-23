import torch
import numpy as np
from .radon import Radon, IRadon

def radon(imten, angles):
    '''
        Compute forward radon operation
        conda install conda-forge::kornia
        Inputs:
            imten: (1, nimg, H, W) image tensor
            angles: (nangles) angles tensor -- should be on same device as
                imten
        Outputs:
            sinogram: (nimg, nangles, W) sinogram
    '''
    import kornia
    nangles = len(angles)
    imten_rep = torch.repeat_interleave(imten, nangles, 0)
    imten_rot = kornia.geometry.rotate(imten_rep, angles)
    sinogram = imten_rot.sum(-1).permute(1, 2, 0)

    return sinogram


def inverse_radon(sinogram):
    return sinogram

class CT_SV():
    def __init__(self, radon_view):
        self.theta = np.linspace(0, 180, radon_view, endpoint=False)

    def A(self, x):
        img_width=x.shape[-1]
        self.radon = Radon(img_width, self.theta, circle=False).to(x.device)
        self.iradon = IRadon(img_width, self.theta, circle=False).to(x.device)
        # return self.iradon(self.radon(x))
        return torch.clamp(self.iradon(self.radon(x)), min=0, max=1)

    def sino(self, x):
        img_width=x.shape[-1]
        self.radon = Radon(img_width, self.theta, circle=False).to(x.device)
        return self.radon(x)

    # def A_pinv(self, x):
    #     return self.A(x)


class CT_LA():
    """
    Limited Angle tomography
    """
    def __init__(self, radon_angle):
        self.theta = torch.arange(radon_angle)

    def A(self, x):
        img_width=x.shape[-1]
        self.radon = Radon(img_width, self.theta, circle=False).to(x.device)
        self.iradon = IRadon(img_width, self.theta, circle=False).to(x.device)
        return torch.clamp(self.iradon(self.radon(x)),min=0,max=1)


    def sino(self, x):
        img_width=x.shape[-1]
        self.radon = Radon(img_width, self.theta, circle=False).to(x.device)
        return self.radon(x)

    # def A_pinv(self, x):
    #     return self.A(x)
