# Grounded-Segment-Anything
Demo combining [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) and [Segment Anything](https://github.com/facebookresearch/segment-anything)! Right now, this is just a simple small project. We will continue to improve it and create more interesting demos.

- [Segment Anything](https://github.com/facebookresearch/segment-anything) is a strong segmentation model. But it needs prompts (like boxes/points) to generate masks
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) is a strong zero-shot detector which is capable of to generate high quality boxes and labels with free-form text
- The combination of the two models enable to **detect and segment everything** with text inputs
- The combination of `BLIP + GroundingDINO + SAM` for automatic labeling
- The combination of `GroundingDINO + SAM + Stable-diffusion` for data-factory, generating new data

**Grounded-SAM + Mask Segregation + Save to database**

<img src="https://github.com/lucyellu/Grounded-SAM/assets/20881728/1b40f242-c644-4a5f-8a15-0d42de129656" width="25%" alt="Image description">

<img src="https://github.com/lucyellu/Grounded-SAM/assets/20881728/01814f02-3183-4030-acfa-1dcac14f784b" width="25%" alt="Image description">

<img src="https://github.com/lucyellu/Grounded-SAM/assets/20881728/7e72bc50-395d-4db3-be92-f47279b60172" width="25%" alt="Image description">



### Efficient SAMs
List of Efficient SAM variants:

<div align="center">

| Title | Intro | Description | Links |
|:----:|:----:|:----:|:----:|
| [FastSAM](https://arxiv.org/pdf/2306.12156.pdf) | ![](https://github.com/CASIA-IVA-Lab/FastSAM/blob/main/assets/Overview.png) | The Fast Segment Anything Model(FastSAM) is a CNN Segment Anything Model trained by only 2% of the SA-1B dataset published by SAM authors. The FastSAM achieve a comparable performance with the SAM method at 50× higher run-time speed. | [[Github](https://github.com/CASIA-IVA-Lab/FastSAM)]  [[Demo](https://huggingface.co/spaces/An-619/FastSAM)] |
| [MobileSAM](https://arxiv.org/pdf/2306.14289.pdf) | ![](https://github.com/ChaoningZhang/MobileSAM/blob/master/assets/model_diagram.jpg?raw=true) | MobileSAM performs on par with the original SAM (at least visually) and keeps exactly the same pipeline as the original SAM except for a change on the image encoder. Specifically, we replace the original heavyweight ViT-H encoder (632M) with a much smaller Tiny-ViT (5M). On a single GPU, MobileSAM runs around 12ms per image: 8ms on the image encoder and 4ms on the mask decoder. | [[Github](https://github.com/ChaoningZhang/MobileSAM)] |
| [Light-HQSAM](https://arxiv.org/pdf/2306.01567.pdf) | ![](https://github.com/SysCV/sam-hq/blob/main/figs/sam-hf-framework.png?raw=true) | Light HQ-SAM is based on the tiny vit image encoder provided by MobileSAM. We design a learnable High-Quality Output Token, which is injected into SAM's mask decoder and is responsible for predicting the high-quality mask. Instead of only applying it on mask-decoder features, we first fuse them with ViT features for improved mask details. Refer to [Light HQ-SAM vs. MobileSAM](https://github.com/SysCV/sam-hq#light-hq-sam-vs-mobilesam-on-coco) for more details. | [[Github](https://github.com/SysCV/sam-hq)] |
| [Efficient-SAM](https://github.com/yformer/EfficientSAM) | ![](https://yformer.github.io/efficient-sam/EfficientSAM_files/overview.png) |Segment Anything Model (SAM) has emerged as a powerful tool for numerous vision applications. However, the huge computation cost of SAM model has limited its applications to wider real-world applications. To address this limitation, we propose EfficientSAMs, light-weight SAM models that exhibit decent performance with largely reduced complexity. Our idea is based on leveraging masked image pretraining, SAMI, which learns to reconstruct features from SAM image encoder for effective visual representation learning. Further, we take SAMI-pretrained light-weight image encoders and mask decoder to build EfficientSAMs, and finetune the models on SA-1B for segment anything task. Refer to [EfficientSAM arXiv](https://arxiv.org/pdf/2312.00863.pdf) for more details.| [[Github](https://github.com/yformer/EfficientSAM)] |
| [Edge-SAM](https://github.com/chongzhou96/EdgeSAM) | ![](https://www.mmlab-ntu.com/project/edgesam/img/arch.png) | EdgeSAM involves distilling the original ViT-based SAM image encoder into a purely CNN-based architecture, better suited for edge devices. We carefully benchmark various distillation strategies and demonstrate that task-agnostic encoder distillation fails to capture the full knowledge embodied in SAM. Refer to [Edge-SAM arXiv](https://arxiv.org/abs/2312.06660) for more details. | [[Github](https://github.com/chongzhou96/EdgeSAM)] |
| [RepViT-SAM](https://github.com/THU-MIG/RepViT/tree/main/sam) | ![](https://jameslahm.github.io/repvit-sam/static/images/edge.png) | Recently, RepViT achieves the state-of-the-art performance and latency trade-off on mobile devices by incorporating efficient architectural designs of ViTs into CNNs. Here, to achieve real-time segmenting anything on mobile devices, following MobileSAM, we replace the heavyweight image encoder in SAM with RepViT model, ending up with the RepViT-SAM model. Extensive experiments show that RepViT-SAM can enjoy significantly better zero-shot transfer capability than MobileSAM, along with nearly 10× faster inference speed. Refer to [RepViT-SAM arXiv](https://arxiv.org/pdf/2312.05760.pdf) for more details. | [[Github](https://github.com/THU-MIG/RepViT)] |





<div align="left">
code mostly from: yizhangliu/Grounded-Segment-Anything


Configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


