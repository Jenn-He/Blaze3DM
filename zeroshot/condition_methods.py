import os
from abc import ABC, abstractmethod
import torch

__CONDITIONING_METHOD__ = {}

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

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, decoder, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, decoder=decoder,**kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, decoder, **kwargs):
        self.operator = operator
        self.noiser = noiser
        self.decoder = decoder

    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        '''if task is SV-CT, LA-CT, CS-MRI:'''
        import random
        sample_num = kwargs['sample_num']
        N, C, H, W = measurement.shape
        xyz = make_coord((N, H, W), flatten=False).flip(-1).unsqueeze(0)
        norm = 0
        rand_z = random.sample(range(0, N), sample_num)
        xyz_batch = xyz[:, rand_z, :, :, :].reshape((1, -1, 3)).to('cuda')
        image_0_hat = self.decoder.decoder(x_0_hat, xyz_batch).reshape(sample_num, 1, H, W)
        y = measurement[rand_z, :, :, :] / 255.0
        ADx = self.operator(image_0_hat)
        norm += torch.linalg.norm(y - ADx)
        norm /= sample_num
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        return norm_grad, norm, y[0, 0, :, :].detach().cpu().numpy(), ADx[0, 0, :, :].detach().cpu().numpy()

        # '''if task is ZSR-MRI:'''
        # import random
        # sample_num = kwargs['sample_num']
        # N, C, H, W = measurement.shape
        # xyz = make_coord((N, H, W), flatten=False).flip(-1).unsqueeze(0)
        # norm=0
        # rand_y=random.sample(range(0,H),sample_num)
        # xyz_batch = xyz[:, :, rand_y, :, :].reshape((1, -1, 3)).to('cuda')
        # image_0_hat = self.decoder.decoder(x_0_hat, xyz_batch).reshape(N, 1, sample_num, W)
        # y=measurement[:, :, rand_y, :]/255.0
        # ADx=self.operator(image_0_hat)
        # norm += torch.linalg.norm(y-ADx)
        # norm/=sample_num
        # norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        # return norm_grad, norm, y[:, 0, 0, :].detach().cpu().numpy(), ADx[:, 0, 0, :].detach().cpu().numpy() # ZAxis
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm
        
@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, decoder,**kwargs):
        super().__init__(operator, noiser, decoder)
        self.scale = kwargs.get('scale', 1.0)
        self.sample_num=kwargs.get('sample_num', 1)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm, y, ADx = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **{'sample_num':self.sample_num})
        x_t -= norm_grad * self.scale
        return x_t, norm, y, ADx
        
@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm


# Implement of Diffusion with Spherical Gaussian Constraint(DSG)
@register_conditioning_method(name='DSG')
class DSG(ConditioningMethod):
    def __init__(self, operator, noiser, decoder, **kwargs):
        super().__init__(operator, noiser, decoder)
        self.interval = kwargs.get('interval', 1) # 10,1!!
        self.guidance_scale = kwargs.get('guidance_scale', 0.1) # 0.2!!,0.1
        print(f'interval: {self.interval}')
        print(f'guidance_scale: {self.guidance_scale}')
        self.sample_num=kwargs.get('sample_num', 1)

    def conditioning(self, x_prev, x_t, x_t_mean, x_0_hat, measurement, idx, **kwargs):
        norm_grad, norm, y, ADx = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement,
                                                      **{'sample_num': self.sample_num})
        if idx % self.interval == 0:
            eps = 1e-8
            norm_grad_norm = torch.linalg.norm(norm_grad) # dim=[1, 2, 3]

            b, c, h, w = x_t.shape
            r = torch.sqrt(torch.tensor(c * h * w)) * kwargs.get('sigma_t', 1.)[0, 0, 0, 0]
            guidance_rate = self.guidance_scale

            d_star = -r * norm_grad / (norm_grad_norm + eps)
            d_sample = x_t - x_t_mean
            mix_direction = d_sample + guidance_rate * (d_star - d_sample)
            mix_direction_norm = torch.linalg.norm(mix_direction) # , dim=[1, 2, 3]
            mix_step = mix_direction / (mix_direction_norm + eps) * r

            x_t = x_t_mean + mix_step

        return x_t, norm, y, ADx