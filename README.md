# Pixel2Mesh++: 3D Mesh Generation and Refinement from Multi-View Images (TPAMI2022)

[Chao Wen⋆](https://walsvid.github.io/),
[Yinda Zhang⋆](https://www.zhangyinda.com/),
[Chenjie Cao](https://github.com/ewrfcas),
[Zhuwen Li](https://scholar.google.com.sg/citations?user=gIBLutQAAAAJ&hl=en),
[Xiangyang Xue](https://scholar.google.com.hk/citations?user=DTbhX6oAAAAJ&hl=zh-CN),
[Yanwei Fu](http://yanweifu.github.io/)


[Project Page](https://ewrfcas.github.io/Pixel2MeshPlusPlus-MVDISN/)

## Abstract

We study the problem of shape generation in 3D mesh representation from a small number of color images with or without camera poses.
While many previous works learn to hallucinate the shape directly from priors, we adopt to further improve the shape quality by
leveraging cross-view information with a graph convolution network. Instead of building a direct mapping function from images to
3D shape, our model learns to predict series of deformations to improve a coarse shape iteratively. Inspired by traditional multiple
view geometry methods, our network samples nearby area around the initial mesh's vertex locations and reasons an optimal deformation
using perceptual feature statistics built from multiple input images. Extensive experiments show that our model produces accurate 3D
shapes that are not only visually plausible from the input perspectives, but also well aligned to arbitrary viewpoints. With the help
of physically driven architecture, our model also exhibits generalization capability across different semantic categories, and the
number of input images. Model analysis experiments show that our model is robust to the quality of the initial mesh and the error of
camera pose, and can be combined with a differentiable renderer for test-time optimization.

## Important

This project only supports training/testing MV-DISN in TPAMI2022 P2M++. More details about training the deformation network in P2M++ can be found in [here](https://github.com/walsvid/Pixel2MeshPlusPlus).

## Data download

We provide the [OneDrive](https://1drv.ms/f/s!Arld9Vmkf6wRgdg8ZcdNvR98vjpSaQ) link to download data.

## Preparation

Compiling for marchingcubes and chamferloss.

```
python setup.py install
```

## Training

Single GPU training:

```
CUDA_VISIBLE_DEVICES=0 python train.py --options ./configs/disn/disn_config.yml
```

Multi-GPU training:

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --options ./configs/disn/disn_config.yml
```

## Testing

```
CUDA_VISIBLE_DEVICES=0 python predict.py --options ./configs/disn/disn_config.yml
```

This project is built upon [Generation3D](https://github.com/walsvid/Generation3D) and [P2M++](https://github.com/walsvid/Pixel2MeshPlusPlus).
