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





duplicated_from: yizhangliu/Grounded-Segment-Anything

Configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
