
import warnings
warnings.filterwarnings('ignore')

import subprocess, io, os, sys, time
from loguru import logger

# os.system("pip install diffuser==0.6.0")
# os.system("pip install transformers==4.29.1")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if os.environ.get('IS_MY_DEBUG') is None:
    result = subprocess.run(['pip', 'install', '-e', 'GroundingDINO'], check=True)
    print(f'pip install GroundingDINO = {result}')

result = subprocess.run(['pip', 'list'], check=True)
print(f'pip list = {result}')

sys.path.insert(0, './GroundingDINO')

if not os.path.exists('./sam_vit_h_4b8939.pth'):
    logger.info(f"get sam_vit_h_4b8939.pth...")
    result = subprocess.run(['wget', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'], check=True)
    print(f'wget sam_vit_h_4b8939.pth result = {result}')    

import gradio as gr

import argparse
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

import cv2
import numpy as np
import matplotlib.pyplot as plt
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config as lama_Config

# segment anything
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator

# diffusers
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
from huggingface_hub import hf_hub_download

from  utils import computer_info

def load_model_hf(model_config_path, repo_id, filename, device='cpu'):
    args = SLConfig.fromfile(model_config_path) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model    

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        font = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
        font_size = 36
        new_font = ImageFont.truetype(font, font_size)

        draw.text((x0+2, y0+2), str(label), font=new_font, fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

def load_image(image_path):
    # # load image
    if isinstance(image_path, PIL.Image.Image):
        image_pil = image_path
    else:
        image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location=device) #"cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

def xywh_to_xyxy(box, sizeW, sizeH):
    if isinstance(box, list):
        box = torch.Tensor(box)
    box = box * torch.Tensor([sizeW, sizeH, sizeW, sizeH])
    box[:2] -= box[2:] / 2
    box[2:] += box[:2]
    box = box.numpy()
    return box

def mask_extend(img, box, extend_pixels=10, useRectangle=True):
    box[0] = int(box[0])
    box[1] = int(box[1])
    box[2] = int(box[2])
    box[3] = int(box[3])
    region = img.crop(tuple(box))
    new_width = box[2] - box[0] + 2*extend_pixels
    new_height = box[3] - box[1] + 2*extend_pixels

    region_BILINEAR = region.resize((int(new_width), int(new_height)))
    if useRectangle:
        region_draw = ImageDraw.Draw(region_BILINEAR)
        region_draw.rectangle((0, 0, new_width, new_height), fill=(255, 255, 255))    
    img.paste(region_BILINEAR, (int(box[0]-extend_pixels), int(box[1]-extend_pixels)))
    return img

def mix_masks(imgs):
    re_img =  1 - np.asarray(imgs[0].convert("1"))
    for i in range(len(imgs)-1):
        re_img = np.multiply(re_img, 1 - np.asarray(imgs[i+1].convert("1")))
    re_img =  1 - re_img
    return  Image.fromarray(np.uint8(255*re_img))

config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swint_ogc.pth"
sam_checkpoint = './sam_vit_h_4b8939.pth' 
output_dir = "outputs"
device = evice = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'device={device}')

# make dir
os.makedirs(output_dir, exist_ok=True)

# initialize groundingdino model
logger.info(f"initialize groundingdino model...")
groundingdino_model = load_model_hf(config_file, ckpt_repo_id, ckpt_filenmae)

# initialize SAM
logger.info(f"initialize SAM model...")
sam_device = device
sam_model = build_sam(checkpoint=sam_checkpoint).to(sam_device)
sam_predictor = SamPredictor(sam_model)
sam_mask_generator = SamAutomaticMaskGenerator(sam_model)

os.system("pip list")
# initialize stable-diffusion-inpainting
logger.info(f"initialize stable-diffusion-inpainting...")
sd_pipe = None
if os.environ.get('IS_MY_DEBUG') is None:
    sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", 
            # revision="fp16",
            # "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
    )
    sd_pipe = sd_pipe.to(device)

# initialize lama_cleaner
logger.info(f"initialize lama_cleaner...")
from lama_cleaner.helper import (
    load_img,
    numpy_to_bytes,
    resize_max_size,
)

lama_cleaner_model = ModelManager(
        name='lama',
        device='cpu', # device,
    )

def lama_cleaner_process(image, mask):
    ori_image = image
    if mask.shape[0] == image.shape[1] and mask.shape[1] == image.shape[0] and mask.shape[0] != mask.shape[1]:
        # rotate image
        ori_image = np.transpose(image[::-1, ...][:, ::-1], axes=(1, 0, 2))[::-1, ...]
        image = ori_image
    
    original_shape = ori_image.shape
    interpolation = cv2.INTER_CUBIC
    
    size_limit = 1080
    if size_limit == "Original":
        size_limit = max(image.shape)
    else:
        size_limit = int(size_limit)

    config = lama_Config(
        ldm_steps=25,
        ldm_sampler='plms',
        zits_wireframe=True,
        hd_strategy='Original',
        hd_strategy_crop_margin=196,
        hd_strategy_crop_trigger_size=1280,
        hd_strategy_resize_limit=2048,
        prompt='',
        use_croper=False,
        croper_x=0,
        croper_y=0,
        croper_height=512,
        croper_width=512,
        sd_mask_blur=5,
        sd_strength=0.75,
        sd_steps=50,
        sd_guidance_scale=7.5,
        sd_sampler='ddim',
        sd_seed=42,
        cv2_flag='INPAINT_NS',
        cv2_radius=5,
    )
    
    if config.sd_seed == -1:
        config.sd_seed = random.randint(1, 999999999)

    # logger.info(f"Origin image shape_0_: {original_shape} / {size_limit}")
    image = resize_max_size(image, size_limit=size_limit, interpolation=interpolation)
    # logger.info(f"Resized image shape_1_: {image.shape}")
    
    # logger.info(f"mask image shape_0_: {mask.shape} / {type(mask)}")
    mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)
    # logger.info(f"mask image shape_1_: {mask.shape} / {type(mask)}")

    res_np_img = lama_cleaner_model(image, mask, config)
    torch.cuda.empty_cache()
  
    image = Image.open(io.BytesIO(numpy_to_bytes(res_np_img, 'png')))
    return  image

# relate anything
from ram_utils import iou, sort_and_deduplicate, relation_classes, MLP, show_anns, ram_show_mask
from ram_train_eval import RamModel,RamPredictor
from mmengine.config import Config as mmengine_Config
input_size = 512
hidden_size = 256
num_classes = 56

# load ram model
model_path = "./checkpoints/ram_epoch12.pth"
ram_config = dict(
    model=dict(
        pretrained_model_name_or_path='bert-base-uncased',
        load_pretrained_weights=False,
        num_transformer_layer=2,
        input_feature_size=256,
        output_feature_size=768,
        cls_feature_size=512,
        num_relation_classes=56,
        pred_type='attention',
        loss_type='multi_label_ce',
    ),
    load_from=model_path,
)
ram_config = mmengine_Config(ram_config)

class Ram_Predictor(RamPredictor):
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = torch.device(device)
        self._build_model()

    def _build_model(self):
        self.model = RamModel(**self.config.model).to(self.device)
        if self.config.load_from is not None:
            self.model.load_state_dict(torch.load(self.config.load_from, map_location=self.device))
        self.model.train()

ram_model = Ram_Predictor(ram_config, device)

# visualization
def draw_selected_mask(mask, draw):
    color = (255, 0, 0, 153)
    nonzero_coords = np.transpose(np.nonzero(mask))
    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)

def draw_object_mask(mask, draw):
    color = (0, 0, 255, 153)
    nonzero_coords = np.transpose(np.nonzero(mask))
    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)

def create_title_image(word1, word2, word3, width, font_path='./assets/OpenSans-Bold.ttf'):
    # Define the colors to use for each word
    color_red = (255, 0, 0)
    color_black = (0, 0, 0)
    color_blue = (0, 0, 255)

    # Define the initial font size and spacing between words
    font_size = 40

    # Create a new image with the specified width and white background
    image = Image.new('RGB', (width, 60), (255, 255, 255))

    # Load the specified font
    font = ImageFont.truetype(font_path, font_size)

    # Keep increasing the font size until all words fit within the desired width
    while True:
        # Create a draw object for the image
        draw = ImageDraw.Draw(image)
        
        word_spacing = font_size / 2
        # Draw each word in the appropriate color
        x_offset = word_spacing
        draw.text((x_offset, 0), word1, color_red, font=font)
        x_offset += font.getsize(word1)[0] + word_spacing
        draw.text((x_offset, 0), word2, color_black, font=font)
        x_offset += font.getsize(word2)[0] + word_spacing
        draw.text((x_offset, 0), word3, color_blue, font=font)
        
        word_sizes = [font.getsize(word) for word in [word1, word2, word3]]
        total_width = sum([size[0] for size in word_sizes]) + word_spacing * 3

        # Stop increasing font size if the image is within the desired width
        if total_width <= width:
            break
            
        # Increase font size and reset the draw object
        font_size -= 1
        image = Image.new('RGB', (width, 50), (255, 255, 255))
        font = ImageFont.truetype(font_path, font_size)
        draw = None

    return image

def concatenate_images_vertical(image1, image2):
    # Get the dimensions of the two images
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Create a new image with the combined height and the maximum width
    new_image = Image.new('RGBA', (max(width1, width2), height1 + height2))

    # Paste the first image at the top of the new image
    new_image.paste(image1, (0, 0))

    # Paste the second image below the first image
    new_image.paste(image2, (0, height1))

    return new_image

def relate_anything(input_image, k):    
    logger.info(f'relate_anything_1_{input_image.size}_')
    w, h = input_image.size
    max_edge = 1500
    if w > max_edge or h > max_edge:
        ratio = max(w, h) / max_edge
        new_size = (int(w / ratio), int(h / ratio))
        input_image.thumbnail(new_size)
    
    logger.info(f'relate_anything_2_')
    # load image
    pil_image = input_image.convert('RGBA')
    image = np.array(input_image)
    sam_masks = sam_mask_generator.generate(image)
    filtered_masks = sort_and_deduplicate(sam_masks)

    logger.info(f'relate_anything_3_')
    feat_list = []
    for fm in filtered_masks:
        feat = torch.Tensor(fm['feat']).unsqueeze(0).unsqueeze(0).to(device)
        feat_list.append(feat)
    feat = torch.cat(feat_list, dim=1).to(device)
    matrix_output, rel_triplets = ram_model.predict(feat)

    logger.info(f'relate_anything_4_')
    pil_image_list = []
    for i, rel in enumerate(rel_triplets[:k]):
        s,o,r = int(rel[0]),int(rel[1]),int(rel[2])
        relation = relation_classes[r]

        mask_image = Image.new('RGBA', pil_image.size, color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_image)
            
        draw_selected_mask(filtered_masks[s]['segmentation'], mask_draw)
        draw_object_mask(filtered_masks[o]['segmentation'], mask_draw)

        current_pil_image = pil_image.copy()
        current_pil_image.alpha_composite(mask_image)
        
        title_image = create_title_image('Red', relation, 'Blue', current_pil_image.size[0])
        concate_pil_image = concatenate_images_vertical(current_pil_image, title_image)
        pil_image_list.append(concate_pil_image)

    logger.info(f'relate_anything_5_{len(pil_image_list)}')
    return pil_image_list

mask_source_draw = "draw a mask on input image"
mask_source_segment = "type what to detect below"

def run_anything_task(input_image, text_prompt, task_type, inpaint_prompt, box_threshold, text_threshold, 
            iou_threshold, inpaint_mode, mask_source_radio, remove_mode, remove_mask_extend, num_relation):
    if (task_type == 'relate anything'):
        output_images = relate_anything(input_image['image'], num_relation)
        return output_images, gr.Gallery.update(label='relate images')

    text_prompt = text_prompt.strip()
    if not ((task_type == 'inpainting' or task_type == 'remove') and mask_source_radio == mask_source_draw):
        if text_prompt == '':
            return [], gr.Gallery.update(label='Detection prompt is not found!😂😂😂😂')

    if input_image is None:
            return [], gr.Gallery.update(label='Please upload a image!😂😂😂😂')

    file_temp = int(time.time())
    logger.info(f'run_anything_task_[{file_temp}]_{task_type}/{inpaint_mode}/[{mask_source_radio}]/{remove_mode}/{remove_mask_extend}_[{text_prompt}]/[{inpaint_prompt}]___1_')

    # load image
    input_mask_pil = input_image['mask']
    input_mask = np.array(input_mask_pil.convert("L"))  
     
    image_pil, image = load_image(input_image['image'].convert("RGB"))

    size = image_pil.size

    output_images = []
    output_images.append(input_image['image'])
    # run grounding dino model
    if (task_type == 'inpainting' or task_type == 'remove') and mask_source_radio == mask_source_draw:
        pass
    else:
        groundingdino_device = 'cpu'
        if device != 'cpu':
            try:
                from groundingdino import _C
                groundingdino_device = 'cuda:0'
            except:
                warnings.warn("Failed to load custom C++ ops. Running on CPU mode Only in groundingdino!")

        groundingdino_device = 'cpu'
        boxes_filt, pred_phrases = get_grounding_output(
            groundingdino_model, image, text_prompt, box_threshold, text_threshold, device=groundingdino_device
        )
        if boxes_filt.size(0) == 0:
            logger.info(f'run_anything_task_[{file_temp}]_{task_type}_[{text_prompt}]_1_[No objects detected, please try others.]_')
            return [], gr.Gallery.update(label='No objects detected, please try others.😂😂😂😂')
        boxes_filt_ori = copy.deepcopy(boxes_filt)

        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }
        image_with_box = plot_boxes_to_image(copy.deepcopy(image_pil), pred_dict)[0]
        output_images.append(image_with_box)

    logger.info(f'run_anything_task_[{file_temp}]_{task_type}_2_')
    if task_type == 'segment' or ((task_type == 'inpainting' or task_type == 'remove') and mask_source_radio == mask_source_segment):
        image = np.array(input_image['image'])
        sam_predictor.set_image(image)

        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.to(sam_device)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2])

        masks, _, _, _ = sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )
        # masks: [9, 1, 512, 512]
        assert sam_checkpoint, 'sam_checkpoint is not found!'
        # draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.cpu().numpy(), plt.gca(), label)
        plt.axis('off')
        image_path = os.path.join(output_dir, f"grounding_seg_output_{file_temp}.jpg")
        plt.savefig(image_path, bbox_inches="tight")
        segment_image_result = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        os.remove(image_path)
        output_images.append(segment_image_result)        

    logger.info(f'run_anything_task_[{file_temp}]_{task_type}_3_')
    if task_type == 'detection' or task_type == 'segment':
        logger.info(f'run_anything_task_[{file_temp}]_{task_type}_9_')
        return output_images, gr.Gallery.update(label='result images')
    elif task_type == 'inpainting' or task_type == 'remove':
        if inpaint_prompt.strip() == '' and mask_source_radio == mask_source_segment:
            task_type = 'remove'

        logger.info(f'run_anything_task_[{file_temp}]_{task_type}_4_')  
        if mask_source_radio == mask_source_draw:
            mask_pil = input_mask_pil
            mask = input_mask          
        else:
            masks_ori = copy.deepcopy(masks)
            if inpaint_mode == 'merge':
                masks = torch.sum(masks, dim=0).unsqueeze(0)
                masks = torch.where(masks > 0, True, False)
            mask = masks[0][0].cpu().numpy()
            mask_pil = Image.fromarray(mask)   
        output_images.append(mask_pil.convert("RGB"))

        if task_type == 'inpainting':
            # inpainting pipeline
            image_source_for_inpaint = image_pil.resize((512, 512))
            image_mask_for_inpaint = mask_pil.resize((512, 512))
            image_inpainting = sd_pipe(prompt=inpaint_prompt, image=image_source_for_inpaint, mask_image=image_mask_for_inpaint).images[0]
        else:
            # remove from mask
            if mask_source_radio == mask_source_segment:
                mask_imgs = []
                masks_shape = masks_ori.shape        
                boxes_filt_ori_array = boxes_filt_ori.numpy()
                if inpaint_mode == 'merge':
                    extend_shape_0 = masks_shape[0]
                    extend_shape_1 = masks_shape[1]
                else:
                    extend_shape_0 = 1
                    extend_shape_1 = 1
                for i in range(extend_shape_0):
                    for j in range(extend_shape_1):                
                        mask = masks_ori[i][j].cpu().numpy()
                        mask_pil = Image.fromarray(mask)
                    
                        if remove_mode == 'segment':
                            useRectangle = False
                        else:
                            useRectangle = True

                        try:
                            remove_mask_extend = int(remove_mask_extend)
                        except:
                            remove_mask_extend = 10
                        mask_pil_exp = mask_extend(copy.deepcopy(mask_pil).convert("RGB"), 
                                        xywh_to_xyxy(torch.tensor(boxes_filt_ori_array[i]), size[0], size[1]),
                                        extend_pixels=remove_mask_extend, useRectangle=useRectangle)
                        mask_imgs.append(mask_pil_exp)
                mask_pil = mix_masks(mask_imgs)
                output_images.append(mask_pil.convert("RGB"))               
            image_inpainting = lama_cleaner_process(np.array(image_pil), np.array(mask_pil.convert("L")))

        image_inpainting = image_inpainting.resize((image_pil.size[0], image_pil.size[1]))
        output_images.append(image_inpainting)
        return output_images, gr.Gallery.update(label='result images')        
    else:
        logger.info(f"task_type:{task_type} error!")
    logger.info(f'run_anything_task_[{file_temp}]_9_9_')
    return output_images, gr.Gallery.update(label='result images')

def change_radio_display(task_type, mask_source_radio):
    text_prompt_visible = True
    inpaint_prompt_visible = False
    mask_source_radio_visible = False
    num_relation_visible = False
    if task_type == "inpainting":
        inpaint_prompt_visible = True
    if task_type == "inpainting" or task_type == "remove":
        mask_source_radio_visible = True   
        if mask_source_radio == mask_source_draw:
            text_prompt_visible = False
    if task_type == "relate anything":
        text_prompt_visible = False
        num_relation_visible = True
    return  gr.Textbox.update(visible=text_prompt_visible), gr.Textbox.update(visible=inpaint_prompt_visible), gr.Radio.update(visible=mask_source_radio_visible), gr.Slider.update(visible=num_relation_visible) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded SAM demo", add_help=True)
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--share", action="store_true", help="share the app")
    args = parser.parse_args()

    print(f'args = {args}')

    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', elem_id="image_upload", tool='sketch', type='pil', label="Upload")    
                task_type = gr.Radio(["detection", "segment", "inpainting", "remove", "relate anything"],  value="detection", 
                                                label='Task type', visible=True) 
                mask_source_radio = gr.Radio([mask_source_draw, mask_source_segment], 
                                    value=mask_source_segment, label="Mask from",
                                    visible=False) 
                text_prompt = gr.Textbox(label="Detection Prompt[To detect multiple objects, seperating each name with '.', like this: cat . dog . chair ]", placeholder="Cannot be empty")                                                
                inpaint_prompt = gr.Textbox(label="Inpaint Prompt (if this is empty, then remove)", visible=False)
                num_relation = gr.Slider(label="How many relations do you want to see", minimum=1, maximum=20, value=5, step=1, visible=False)
                run_button = gr.Button(label="Run", visible=True)
                with gr.Accordion("Advanced options", open=False) as advanced_options:
                    box_threshold = gr.Slider(
                        label="Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.001
                    )
                    text_threshold = gr.Slider(
                        label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                    )
                    iou_threshold = gr.Slider(
                        label="IOU Threshold", minimum=0.0, maximum=1.0, value=0.8, step=0.001
                    )                    
                    inpaint_mode = gr.Radio(["merge", "first"], value="merge", label="inpaint_mode")
                    with gr.Row():
                        with gr.Column(scale=1):
                            remove_mode = gr.Radio(["segment", "rectangle"],  value="segment", label='remove mode') 
                        with gr.Column(scale=1):
                            remove_mask_extend = gr.Textbox(label="remove_mask_extend", value='10')

            with gr.Column():
                image_gallery = gr.Gallery(label="result images", show_label=True, elem_id="gallery", visible=True
                    ).style(preview=True, columns=[5], object_fit="scale-down", height="auto")          

            run_button.click(fn=run_anything_task, inputs=[
                            input_image, text_prompt, task_type, inpaint_prompt, box_threshold, text_threshold, iou_threshold, inpaint_mode, mask_source_radio, remove_mode, remove_mask_extend, num_relation], outputs=[image_gallery, image_gallery], show_progress=True, queue=True)
            
            mask_source_radio.change(fn=change_radio_display, inputs=[task_type, mask_source_radio], outputs=[text_prompt, inpaint_prompt, mask_source_radio, num_relation])
            task_type.change(fn=change_radio_display, inputs=[task_type, mask_source_radio], outputs=[text_prompt, inpaint_prompt, mask_source_radio, num_relation])

        DESCRIPTION = f'### This demo from [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything). <br>'
        DESCRIPTION += f'RAM from [RelateAnything](https://github.com/Luodian/RelateAnything). <br>'
        DESCRIPTION += f'Thanks for their excellent work.'
        DESCRIPTION += f'<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. \
                        <a href="https://huggingface.co/spaces/yizhangliu/Grounded-Segment-Anything?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>'
        gr.Markdown(DESCRIPTION)

    computer_info()
    block.launch(server_name='0.0.0.0', debug=args.debug, share=args.share)
