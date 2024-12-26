import requests
import json
import gradio as gr
import base64
from io import BytesIO
from PIL import Image

from modules.shared import opts
import modules.shared as shared
from modules.ui import plaintext_to_html

def url_requests(url, data):
    resp = requests.post(url, data=json.dumps(data))
    img_strs = json.loads(resp.text)["images"]

    images_list = []
    for img_str in img_strs:
        img_byte = base64.b64decode(img_str)
        img_io = BytesIO(img_byte)  # convert image to file-like object
        img = Image.open(img_io)   # img is now PIL Image object
        images_list.append(img)


    return images_list


def txt2img(id_task: str, request: gr.Request, prompt: str, negative_prompt: str, prompt_styles, n_iter: int, batch_size: int, cfg_scale: float, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float, hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, hr_checkpoint_name: str, hr_sampler_name: str, hr_scheduler: str, hr_prompt: str, hr_negative_prompt, override_settings_texts, *args, force_enable_hr=False):

    # faked progress
    if shared.state.job_count == -1:
        shared.state.job_count = n_iter

    shared.state.sampling_steps = args[1]

    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_images_per_prompt": batch_size,
        "steps": args[1],
        "guidance_scale": cfg_scale,
        "seed": args[7],
        "height": height,
        "width": width,
        "strength": args[6]}

    url = shared.cmd_opts.opea_url 

    images = url_requests(url, data)


    shared.total_tqdm.clear()

    generation_info_js = {"prompt": prompt}
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        images = []

    return images, generation_info_js, plaintext_to_html(prompt), plaintext_to_html(f"Steps: {args[1]}", classname="comments")
