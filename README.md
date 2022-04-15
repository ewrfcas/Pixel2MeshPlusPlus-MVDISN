# Pixel2Mesh++: Multi-View 3D Mesh Generation via Deformation

[Chao Wen⋆](https://github.com/walsvid),
[Yinda Zhang⋆](https://www.zhangyinda.com/),
[Zhuwen Li](https://scholar.google.com.sg/citations?user=gIBLutQAAAAJ&hl=en),
[Xiangyang Xue](https://scholar.google.com.hk/citations?user=DTbhX6oAAAAJ&hl=zh-CN),
[Chenjie Cao](https://github.com/ewrfcas),
[Yanwei Fu](http://yanweifu.github.io/)

[![LICENSE](https://img.shields.io/github/license/ewrfcas/Pixel2MeshPlusPlus-MVDISN)](https://github.com/ewrfcas/Pixel2MeshPlusPlus-MVDISN/blob/main/LICENSE)

[comment]: <> (![teaser]&#40;assets/teaser_new.png&#41;)

[comment]: <> ([arXiv]&#40;https://arxiv.org/abs/2103.15087&#41; | [Project Page]&#40;https://ewrfcas.github.io/MST_inpainting/&#41;)


[comment]: <> (## Overview)

[comment]: <> (![teaser]&#40;assets/overview_new1.png&#41;)

[comment]: <> (We learn an encoder-decoder model, which encodes a Sketch Tensor &#40;ST&#41; space consisted of refined lines and edges.)

[comment]: <> (Then the model recover the masked images by the ST space.)

[comment]: <> (### News)

[comment]: <> (- [x] Release the inference codes.)

[comment]: <> (- [x] Training codes.)

[comment]: <> (**Now, this work has been further improved in [ZITS]&#40;https://github.com/DQiaole/ZITS_inpainting&#41; &#40;CVPR2022&#41;**.)

[comment]: <> ([comment]: <> &#40;- [ ] Release the GUI codes.&#41;)

[comment]: <> (### Preparation)

[comment]: <> (1. Preparing the environment.)

[comment]: <> (2. Download the pretrained masked wireframe detection model [LSM-HAWP]&#40;https://drive.google.com/drive/folders/1yg4Nc20D34sON0Ni_IOezjJCFHXKGWUW?usp=sharing&#41; &#40;retrained from [HAWP CVPR2020]&#40;https://github.com/cherubicXN/hawp&#41;&#41;.)

[comment]: <> (3. Download weights for different requires to the 'check_points' fold.)

[comment]: <> (   [P2M]&#40;https://drive.google.com/drive/folders/1uQAzfYvRIAE-aSpYRJbJo-2vBiwit0TK?usp=sharing&#41; &#40;Man-made Places2&#41;,)

[comment]: <> (   [P2C]&#40;https://drive.google.com/drive/folders/1td0SNBdSdzMdj4Ei_GnMmglFYOgwUcM0?usp=sharing&#41; &#40;Comprehensive Places2&#41;,)

[comment]: <> (   [shanghaitech]&#40;https://drive.google.com/drive/folders/1VsHSRGBpGWjTP-LLZPrtW-DQan3FRnEl?usp=sharing&#41; &#40;[Shanghaitech]&#40;https://github.com/huangkuns/wireframe&#41; with all man-made scenes&#41;.)

[comment]: <> (4. For training, we provide irregular and segmentation masks &#40;[download]&#40;https://drive.google.com/drive/folders/1eU6VaTWGdgCXXWueCXilt6oxHdONgUgf?usp=sharing&#41;&#41; with different masking rates. And you should define the mask file list before the training &#40;flist_example.txt&#41;.)

[comment]: <> (### Training)

[comment]: <> (Since the training code is rewritten, there are some differences compared with the test code.)

[comment]: <> (> 1. Training uses src/models.py while testing uses src/model_inference.py.)

[comment]: <> (>)

[comment]: <> (> 2. Image are valued in -1 to 1 &#40;training&#41; and 0 to 1 &#40;testing&#41;.)

[comment]: <> (>)

[comment]: <> (> 3. Masks are always concated to the inputs.)

[comment]: <> (1. Generating wireframes by lsm-hawp.)

[comment]: <> (```)

[comment]: <> (CUDA_VISIBLE_DEVICES=0 python lsm_hawp_inference.py --ckpt_path <best_lsm_hawp.pth> --input_path <input image path> --output_path <output image path>)

[comment]: <> (```)

[comment]: <> (2. Setting file lists in training_configs/config_MST.yml &#40;example: flist_example.txt&#41;.)

[comment]: <> (3. Train the inpainting model with stage1 and stage2.)

[comment]: <> (```)

[comment]: <> (python train_MST_stage1.py --path <model_name> --config training_configs/config_MST.yml --gpu 0)

[comment]: <> (python train_MST_stage2.py --path <model_name> --config training_configs/config_MST.yml --gpu 0)

[comment]: <> (```)

[comment]: <> (For DDP training with multi-gpus:)

[comment]: <> (```)

[comment]: <> (python -m torch.distributed.launch --nproc_per_node=4 train_MST_stage1.py --path <model_name> --config training_configs/config_MST.yml --gpu 0,1,2,3)

[comment]: <> (python -m torch.distributed.launch --nproc_per_node=4 train_MST_stage2.py --path <model_name> --config training_configs/config_MST.yml --gpu 0,1,2,3)

[comment]: <> (```)

[comment]: <> (### Test for a single image)

[comment]: <> (```)

[comment]: <> (python test_single.py --gpu_id 0 \)

[comment]: <> (                      --PATH ./check_points/MST_P2C \)

[comment]: <> (                      --image_path <your image path> \)

[comment]: <> (                      --mask_path <your mask path &#40;0 means valid and 255 means masked&#41;>)

[comment]: <> (```)

