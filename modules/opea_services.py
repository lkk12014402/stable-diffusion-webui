import requests
import json
import gradio as gr
import base64
from io import BytesIO
from PIL import Image

from modules.shared import opts
import modules.shared as shared
from modules.ui import plaintext_to_html

from modules.infotext_utils import create_override_settings_dict
import modules.images as images_util

def url_requests(url, data, return_base64str=False):
    resp = requests.post(url, data=json.dumps(data))
    img_strs = json.loads(resp.text)["images"]
    if return_base64str:
        return img_strs

    images_list = []
    for img_str in img_strs:
        img_byte = base64.b64decode(img_str)
        img_io = BytesIO(img_byte)  # convert image to file-like object
        img = Image.open(img_io)   # img is now PIL Image object
        images_list.append(img)


    return images_list


def txt2img(id_task: str, request: gr.Request, prompt: str, negative_prompt: str, prompt_styles, n_iter: int, batch_size: int, cfg_scale: float, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float, hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, hr_checkpoint_name: str, hr_sampler_name: str, hr_scheduler: str, hr_prompt: str, hr_negative_prompt, override_settings_texts, *args, force_enable_hr=False):

    print("opea microservice: txt2img.")

    # faked progress
    if shared.state.job_count == -1:
        shared.state.job_count = n_iter

    shared.state.sampling_steps = args[1]

    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_images_per_prompt": batch_size,
        "num_inference_steps": args[1],
        "guidance_scale": cfg_scale,
        "seed": args[7],
        "height": height,
        "width": width,
        "strength": args[6]}

    url = shared.cmd_opts.opea_txt2img_url 

    images = url_requests(url, data)


    shared.total_tqdm.clear()

    generation_info_js = {"prompt": prompt}
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        images = []

    return images, generation_info_js, plaintext_to_html(prompt), plaintext_to_html(f"Steps: {args[1]}", classname="comments")


def img2img(id_task: str, request: gr.Request, mode: int, prompt: str, negative_prompt: str, prompt_styles, init_img, sketch, init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig, init_img_inpaint, init_mask_inpaint, mask_blur: int, mask_alpha: float, inpainting_fill: int, n_iter: int, batch_size: int, cfg_scale: float, image_cfg_scale: float, denoising_strength: float, selected_scale_tab: int, height: int, width: int, scale_by: float, resize_mode: int, inpaint_full_res: bool, inpaint_full_res_padding: int, inpainting_mask_invert: int, img2img_batch_input_dir: str, img2img_batch_output_dir: str, img2img_batch_inpaint_mask_dir: str, override_settings_texts, img2img_batch_use_png_info: bool, img2img_batch_png_info_props: list, img2img_batch_png_info_dir: str, img2img_batch_source_type: str, img2img_batch_upload: list, *args):

    print("opea microservice: img2img.")

    override_settings = create_override_settings_dict(override_settings_texts)

    is_batch = mode == 5

    if mode == 0:  # img2img
        image = init_img
        mask = None
    elif mode == 1:  # img2img sketch
        image = sketch
        mask = None
    elif mode == 2:  # inpaint
        image, mask = init_img_with_mask["image"], init_img_with_mask["mask"]
        mask = processing.create_binary_mask(mask)
    elif mode == 3:  # inpaint sketch
        image = inpaint_color_sketch
        orig = inpaint_color_sketch_orig or inpaint_color_sketch
        pred = np.any(np.array(image) != np.array(orig), axis=-1)
        mask = Image.fromarray(pred.astype(np.uint8) * 255, "L")
        mask = ImageEnhance.Brightness(mask).enhance(1 - mask_alpha / 100)
        blur = ImageFilter.GaussianBlur(mask_blur)
        image = Image.composite(image.filter(blur), orig, mask.filter(blur))
    elif mode == 4:  # inpaint upload mask
        image = init_img_inpaint
        mask = init_mask_inpaint
    else:
        image = None
        mask = None

    image = images_util.fix_image(image)
    mask = images_util.fix_image(mask)

    if selected_scale_tab == 1 and not is_batch:
        assert image, "Can't scale by because no image is selected"

        width = int(image.width * scale_by)
        height = int(image.height * scale_by)

    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

    # faked progress
    if shared.state.job_count == -1:
        shared.state.job_count = n_iter

    shared.state.sampling_steps = args[1]

    buffered = BytesIO()
    image.convert('RGB').save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue())


    data = {
        "image": img_b64.decode(),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_images_per_prompt": batch_size,
        "num_inference_steps": args[8],
        "guidance_scale": cfg_scale,
        "seed": args[14],
        "height": height,
        "width": width,
        "strength": args[13]}

    url = shared.cmd_opts.opea_img2img_url

    images = url_requests(url, data)


    if shared.opts.enable_console_prompts:
        print(f"\nimg2img: {prompt}", file=shared.progress_print_out)


    shared.total_tqdm.clear()

    generation_info_js = {"prompt": prompt}
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        images = []

    return images, generation_info_js, plaintext_to_html(prompt), plaintext_to_html(f"Steps: {args[8]}", classname="comments")

