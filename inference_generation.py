import torch
from torchvision.utils import save_image

from PIL import Image
from lavis.models import load_model_and_preprocess
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(
                    prog='BLIP-Diffusion')
parser.add_argument('ckpt_path', default=None)
args = parser.parse_args()

model, vis_preprocess, txt_preprocess = load_model_and_preprocess("blip_diffusion", "base", device="cuda", is_eval=True)
finetuned_ckpt = args.ckpt_path

model.load_checkpoint(finetuned_ckpt)

cond_subject = "scene"
tgt_subject = "scene"
text_prompt = "raining"
# prompt = "in oil painting"

cond_subjects = [txt_preprocess["eval"](cond_subject)]
tgt_subjects = [txt_preprocess["eval"](tgt_subject)]
text_prompt = [txt_preprocess["eval"](text_prompt)]

samples = {
    "cond_images": None,
    "cond_subject": cond_subjects,
    "tgt_subject": tgt_subjects,
    "prompt": text_prompt,
}

num_output = 4

iter_seed = 8888
guidance_scale = 7.5
num_inference_steps = 50
negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

for i in range(num_output):
    output = model.generate(
        samples,
        seed=iter_seed + i,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        neg_prompt=negative_prompt,
        height=512,
        width=512,
    )

    save_image(output[0], f'generated_{cond_subject}_{tgt_subject}_{i}.png')

