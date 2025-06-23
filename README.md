# Blaze3DM: Integrating Triplane Representation with Diffusion for Solving 3D Inverse Problems in Medical Imaging

This repository  `Blaze3DM` is the official implementation of the paper [Blaze3DM: Integrating Triplane Representation with Diffusion for Solving 3D Inverse Problems in Medical Imaging]. The paper has been accepted by **MICCAI 2025** !

Blaze3DM is a novel approach that enables fast and high-fidelity generation by grating compact triplane neural field and a powerful diffusion model. Extensive experiments on zero-shot 3D medical inverse problem solving, including sparse-view CT, limited-angle CT, compressed-sensing MRI, and MRI isotropic super-resolution, demonstrate that Blaze3DM not only achieves state-of-the-art performance but also markedly improves computational efficiency over existing methods (22~40× faster than previous work).

## Usage

### 1. Installation

Here's a summary of the critical dependencies.

- python 3.10
- pytorch 2.2.0
- CUDA 11.8

We recommend the following demand to install all of the dependencies.

```bash
conda env create -f environment.yml
```

To activate the environment, run:

```bash
conda activate blaze3dm
```

### 2. Triplane Fitting

Given some high-quality medical volumes, we construct their corresponding triplanes for each data. The code for triplane fitting can be found in the `triplane` folder, where `triplane/train.py`  jointly trains the triptane representation of each 3D image in the training dataset and the MLP decoder.

```bash
python triplane/train.py
```

### 3. Diffusion Model Training

After constructing the triplane representation of 3D medical images, we obtained an equal number of triplanes for training the latent diffusion model. The code is provided in the `diffusion` folder, especially the `diffusion/train.py`. 

```bash
python diffusion/train.py
```

### 4. Guidance-based Sampling

With the powerful ability of diffusion models to fit data distribution, we have learned the data distribution of triplane representation of 3D medical images. In diffusion model inference, we input degraded medical images (e.g. SV-CT, LA-CT, CS-MRI, ZSR-MRI) and corresponding degradation formation (e.g. dowmsample sinogram or k-space) to perform zeroshot 3D inverse problem solving. The code is provided in the `zeroshot` folder, run the `sample.py` to perform guidance-based sampling. 

```bash
python sample.py
```

## Acknowledgments

The triplane fitting implementation is based on [NFD](https://github.com/JRyanShue/NFD) and [Sin3DM](https://github.com/Sin3DM/Sin3DM), the latent diffusion model is based on [guided diffusion](https://github.com/openai/guided-diffusion), and the zeroshot sampling is based on [DPS](https://github.com/DPS2022/diffusion-posterior-sampling). Thanks for their work and for sharing the code.


