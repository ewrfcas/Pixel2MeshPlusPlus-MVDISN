# Pixel2Mesh++: Multi-View 3D Mesh Generation via Deformation

[Chao Wen⋆](https://walsvid.github.io/),
[Yinda Zhang⋆](https://www.zhangyinda.com/),
[Chenjie Cao](https://github.com/ewrfcas),
[Zhuwen Li](https://scholar.google.com.sg/citations?user=gIBLutQAAAAJ&hl=en),
[Xiangyang Xue](https://scholar.google.com.hk/citations?user=DTbhX6oAAAAJ&hl=zh-CN),
[Yanwei Fu](http://yanweifu.github.io/)

[![LICENSE](https://img.shields.io/github/license/ewrfcas/Pixel2MeshPlusPlus-MVDISN)](https://github.com/ewrfcas/Pixel2MeshPlusPlus-MVDISN/blob/main/LICENSE)

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

