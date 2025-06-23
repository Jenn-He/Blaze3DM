import torch
import logging
import numpy as np

class MRI_CS():
    def __init__(self, acc_factor: int):
        self.acc_factor=acc_factor

    def A(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape

        assert C == 1
        kspace=torch.fft.fftshift(torch.fft.fft2(x), dim=[-1, -2])
        from sigpy.mri import poisson
        mask = poisson((H, W), accel=self.acc_factor).astype(np.float32)
        mask = torch.from_numpy(mask).repeat(N, 1, 1, 1).to(x.device)
        result = kspace * mask
        result = torch.real(torch.fft.ifft2(torch.fft.ifftshift(result, dim=[-1, -2])))
        result[result<0]=0 # clip the negative value

        return result

    # def A_pinv(self, x:torch.Tensor):
    #     return self.A(x)


class MRI_ZAxis():
    def __init__(self, zsr_factor: int):
        self.factor = zsr_factor
    
    def A(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape

        assert C == 1
        result = torch.zeros((N, C, H, W), device=x.device)
        for i in range(self.factor):
            result[i::self.factor,:,:,:] = x[::self.factor,:,:,:]

        return result
    
    # def A_pinv(self, x:torch.Tensor):
    #     N, C, H, W = x.shape
    #     assert C == 1
    #
    #     x = x.clone().detach()
    #     result = x.repeat_interleave(self.factor, dim=0)
    #
    #     return result