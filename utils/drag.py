import math
import os
import numpy as np
import pickle
from PIL import Image
import cv2
from tqdm import tqdm

import torch
import torch.nn.functional as F

from .cycle_sde import Sampler, get_img_latent, get_text_embed, load_model, set_seed

# select version from v1-5 (recommended), v2-1, xl
sd_version = 'v1-5'
vae, tokenizer, text_encoder, unet, scheduler, feature_extractor, image_encoder, tokenizer_2, text_encoder_2 = load_model(sd_version)

def scale_schedule(begin, end, n, length, type='linear'):
    if type == 'constant':
        return end
    elif type == 'linear':
        return begin + (end - begin) * n / length
    elif type == 'cos':
        factor = (1 - math.cos(n * math.pi / length)) / 2
        return (1 - factor) * begin + factor * end
    else:
        raise NotImplementedError(type)
    
def get_meta_data(meta_data_path):
    with open(meta_data_path, 'rb') as file:
        meta_data = pickle.load(file)
        prompt = meta_data['prompt']
        mask = meta_data['mask']
        points = meta_data['points']
        source = points[0:-1:2]
        target = points[1::2]
    return prompt, mask, source, target

def get_drag_data(data_path):
    ori_image_path = os.path.join(data_path, 'original_image.png')
    meta_data_path = os.path.join(data_path, 'meta_data_region.pkl')

    original_image = np.array(Image.open(ori_image_path))
    prompt, mask, source, target = get_meta_data(meta_data_path)

    return {
        'ori_image' : original_image, 'preview' : original_image, 'prompt' : prompt, 
        'mask' : mask, 'source' : np.array(source), 'target' : np.array(target)
    }

def reverse_and_repeat_every_n_elements(lst, n, repeat=1):
    """
    Reverse every n elements in a given list, then repeat the reversed segments
    the specified number of times.
    Example:
    >>> reverse_and_repeat_every_n_elements([1, 2, 3, 4, 5, 6, 7, 8, 9], 3, 2)
    [3, 2, 1, 3, 2, 1, 6, 5, 4, 6, 5, 4, 9, 8, 7, 9, 8, 7]
    """
    if not lst or n < 1:
        return lst
    return [element for i in range(0, len(lst), n) for _ in range(repeat) for element in reversed(lst[i:i+n])]

def get_border_points(points):
    x_max, y_max = np.amax(points, axis=0) 
    mask = np.zeros((y_max+1, x_max+1), np.uint8)
    mask[points[:, 1], points[:, 0]] = 1
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    border_points = np.concatenate([contour[:, 0, :] for contour in contours], axis=0)
    return border_points

def postprocess(vae, latent, ori_image, mask):
    dtype = latent.dtype
    upcast_dtype = torch.float32 if 'xl-base-1.0' in vae.config._name_or_path and dtype == torch.float16 else dtype
    H, W = ori_image.shape[:2]

    if dtype == torch.float16:
        vae = vae.to(upcast_dtype)
        for module in [vae.post_quant_conv, vae.decoder.conv_in, vae.decoder.mid_block]:
            module = module.to(dtype)
    
    image = vae.decode(latent / 0.18215).sample / 2 + 0.5
    image = (image.clamp(0, 1).permute(0, 2, 3, 1)[0].cpu().numpy() * 255).astype(np.uint8)
    image = cv2.resize(image, (W, H))
    
    if not np.all(mask == 1):
        image = np.where(mask[:, :, None], image, ori_image)
    
    return image

def copy_and_paste(source_latents, target_latents, source, target):
    target_latents[0, :, target[:, 1], target[:, 0]] = source_latents[0, :, source[:, 1], source[:, 0]]
    return target_latents

def blur_source(latents, noise_scale, source):
    img_scale = (1 - noise_scale ** 2) ** (0.5) if noise_scale < 1 else 0
    latents[0, :, source[:, 1], source[:, 0]] = latents[0, :, source[:, 1], source[:, 0]] * img_scale + \
        torch.randn_like(latents[0, :, source[:, 1], source[:, 0]]) * noise_scale
    return latents

def ip_encode_image(feature_extractor, image_encoder, image):
    dtype = next(image_encoder.parameters()).dtype
    device = next(image_encoder.parameters()).device

    image = feature_extractor(image, return_tensors="pt").pixel_values.to(device=device, dtype=dtype)
    image_enc_hidden_states = image_encoder(image, output_hidden_states=True).hidden_states[-2]      
    uncond_image_enc_hidden_states = image_encoder(
        torch.zeros_like(image), output_hidden_states=True
    ).hidden_states[-2]
    image_embeds = torch.stack([uncond_image_enc_hidden_states, image_enc_hidden_states], dim=0)

    return [image_embeds]    
        
def forward(scheduler, sampler, steps, start_t, latent, text_embeddings, added_cond_kwargs, progress=tqdm, sde=True):
    forward_func = sampler.forward_sde if sde else sampler.forward_ode
    hook_latents = [latent,]; noises = []; cfg_scales = []
    start_t = int(start_t * steps)

    for index, t in enumerate(progress(scheduler.timesteps[(steps - start_t):].flip(dims=[0])), start=1):
        cfg_scale = scale_schedule(1, 1, index, steps, type='linear')
        latent, noise = forward_func(t, latent, cfg_scale, text_embeddings, added_cond_kwargs=added_cond_kwargs)
        hook_latents.append(latent); noises.append(noise); cfg_scales.append(cfg_scale)

    return hook_latents, noises, cfg_scales

def backward(scheduler, sampler, steps, start_t, end_t, noise_scale, hook_latents, noises, cfg_scales, mask, text_embeddings, added_cond_kwargs, blur, source, target, progress=tqdm, latent=None, sde=True):
    start_t = int(start_t * steps)
    end_t = int(end_t * steps)

    latent = hook_latents[-1].clone() if latent is None else latent
    latent = blur_source(latent, noise_scale, blur)

    for t in progress(scheduler.timesteps[(steps-start_t- 1):-1]):
        hook_latent = hook_latents.pop()
        latent = copy_and_paste(hook_latent, latent, source, target) if t >= end_t else latent
        latent = torch.where(mask == 1, latent, hook_latent)
        latent = sampler.sample(t, latent, cfg_scales.pop(), text_embeddings, sde=sde, noise=noises.pop(), added_cond_kwargs=added_cond_kwargs)
    return latent

def drag(drag_data, steps, start_t, end_t, noise_scale, seed, progress=tqdm, method='Encode then CP', save_path=''):
    def copy_key_hook(module, input, output):
        keys.append(output)
    def copy_value_hook(module, input, output):
        values.append(output)
    def paste_key_hook(module, input, output):
        output[:] = keys.pop()
    def paste_value_hook(module, input, output):
        output[:] = values.pop()

    def register(do='copy'):
        do_copy = do == 'copy'
        key_hook, value_hook = (copy_key_hook, copy_value_hook) if do_copy else (paste_key_hook, paste_value_hook)
        key_handlers = []; value_handlers = []
        for block in (*sampler.unet.down_blocks, sampler.unet.mid_block, *sampler.unet.up_blocks):
            if not hasattr(block, 'attentions'):
                continue
            for attention in block.attentions:
                for tb in attention.transformer_blocks:
                    key_handlers.append(tb.attn1.to_k.register_forward_hook(key_hook))
                    value_handlers.append(tb.attn1.to_v.register_forward_hook(value_hook))
        return key_handlers, value_handlers

    def unregister(*handlers):
        for handler in handlers:
            handler.remove()
        torch.cuda.empty_cache()

    set_seed(seed)
    device = 'cuda'
    sde = encode_then_cp = method == 'Encode then CP'
    
    ori_image, preview, prompt, mask, source, target = drag_data.values()
    source = torch.from_numpy(source).to(device) if isinstance(source, np.ndarray) else source.to(device)
    target = torch.from_numpy(target).to(device) if isinstance(target, np.ndarray) else target.to(device)
    source = source // 8; target = target // 8 # from img scale to latent scale

    if encode_then_cp:
        blur_pts = source; copy_pts = source
    else:
        blur_pts = torch.cat([torch.from_numpy(get_border_points(target.cpu().numpy())).to(device), source], dim=0)
        copy_pts = target
    paste_pts = target
    
    latent = get_img_latent(ori_image, vae)
    preview_latent = get_img_latent(preview, vae) if not encode_then_cp else None
    sampler = Sampler(unet=unet, scheduler=scheduler, num_steps=steps)

    with torch.no_grad():
        neg_pooled_prompt_embeds, neg_prompt_embeds = get_text_embed("", tokenizer, text_encoder, tokenizer_2, text_encoder_2)
        neg_prompt_embeds = neg_prompt_embeds if sd_version == 'xl' else neg_pooled_prompt_embeds
        pooled_prompt_embeds, prompt_embeds = get_text_embed(prompt, tokenizer, text_encoder, tokenizer_2, text_encoder_2)
        prompt_embeds = prompt_embeds if sd_version == 'xl' else pooled_prompt_embeds
        prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        image_embeds = ip_encode_image(feature_extractor, image_encoder, ori_image)
        
        H, W = ori_image.shape[:2]
        add_time_ids = torch.tensor([[H, W, 0, 0, H, W]]).to(prompt_embeds).repeat(2, 1)
        added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids} if sd_version == 'xl' else {}
        added_cond_kwargs["image_embeds"] = image_embeds        

        mask_pt = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
        mask_pt = F.interpolate(mask_pt, size=latent.shape[2:]).expand_as(latent)

        if not encode_then_cp:
            hook_latents, noises, cfg_scales = forward(scheduler, sampler, steps, start_t, preview_latent, prompt_embeds, added_cond_kwargs, progress=progress, sde=sde)            

        keys = []; values = []
        key_handlers, value_handlers = register(do='copy')
        if encode_then_cp:
            hook_latents, noises, cfg_scales = forward(scheduler, sampler, steps, start_t, latent, prompt_embeds, added_cond_kwargs, progress=progress, sde=sde)
            start_latent = None
        else:
            start_latent = forward(scheduler, sampler, steps, start_t, latent, prompt_embeds, added_cond_kwargs, progress=progress, sde=sde)[0][-1]
        unregister(*key_handlers, *value_handlers)

        keys = reverse_and_repeat_every_n_elements(keys, n=len(key_handlers))
        values = reverse_and_repeat_every_n_elements(values, n=len(value_handlers))

        key_handlers, value_handlers = register(do='paste')
        latent = backward(scheduler, sampler, steps, start_t, end_t, noise_scale, hook_latents, noises, cfg_scales, mask_pt, prompt_embeds, added_cond_kwargs, blur_pts, copy_pts, paste_pts, latent=start_latent, progress=progress, sde=sde)
        unregister(*key_handlers, *value_handlers)

        image = postprocess(vae, latent, ori_image, mask)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        counter = 0
        file_root, file_extension = os.path.splitext('dragged_image.png')
        while True:
            test_name = f"{file_root} ({counter}){file_extension}" if counter != 0 else 'dragged_image.png'
            full_path = os.path.join(save_path, test_name)
            if not os.path.exists(full_path):
                Image.fromarray(image).save(full_path)
                break
            counter += 1
    torch.cuda.empty_cache()
    return image