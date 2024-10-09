import warnings
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn.functional as F

import lpips
from diffusers import StableDiffusionPipeline
from transformers import AutoModel

warnings.filterwarnings(action='ignore', category=UserWarning)

def plot_matching_result(src_image, trg_image, src_points, trg_points, pred_trg_points, output_path=None, border_width=10):
    """
        Used to visualize dragging effect.
    """
    if src_points.shape != trg_points.shape or src_points.shape != pred_trg_points.shape:
        raise ValueError(f"points arrays must have the same shapes, got Source:{src_points.shape}, Target:{trg_points.shape}, Predicted:{pred_trg_points.shape}")
    if src_image.shape[0] != trg_image.shape[0]:
        raise ValueError(f"Source image and target image must have same height, got Source:{src_image.shape}, Target:{trg_image.shape}")

    # Create a border and combine images
    border = np.ones((src_image.shape[0], border_width, 3), dtype=np.uint8) * 255
    combined_image = np.concatenate((src_image, border, trg_image), axis=1)

    # Adjust target and predicted target points by the width of the source image and the border
    trg_points_adj = trg_points + np.array([src_image.shape[1] + border_width, 0])
    pred_trg_points_adj = pred_trg_points + np.array([src_image.shape[1] + border_width, 0])

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(combined_image.astype(np.uint8))

    # Draw arrows and points
    for src_pt, trg_pt, pred_trg_pt in zip(src_points, trg_points_adj, pred_trg_points_adj):
        ax.scatter(src_pt[0], src_pt[1], color='red', s=20)
        ax.scatter(trg_pt[0], trg_pt[1], color='blue', s=20)
        ax.scatter(pred_trg_pt[0], pred_trg_pt[1], color='red', s=20)

        # Draw arrows from src to trg
        ax.arrow(src_pt[0], src_pt[1], trg_pt[0] - src_pt[0], trg_pt[1] - src_pt[1], 
                 head_width=5, head_length=10, fc='white', ec='white', length_includes_head=True, lw=2, alpha=0.8)

        # Draw arrows from pred_trg to trg
        ax.arrow(pred_trg_pt[0], pred_trg_pt[1], trg_pt[0] - pred_trg_pt[0], trg_pt[1] - pred_trg_pt[1], 
                 head_width=5, head_length=10, fc='white', ec='white', length_includes_head=True, lw=2, alpha=0.8)

    ax.axis('off')

    # Save or show the image
    if output_path is not None:
        directory = os.path.dirname(output_path)
        if directory != '':
            os.makedirs(directory, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)  # Save with higher resolution

    plt.show()
    plt.close()

    return fig if output_path is None else output_path


def create_mask(src_points, trg_points, img_size):
    """
    Creates a batch of masks based on the distance of image pixels to batches of given points.

    Args:
        src_points (torch.Tensor): The source points coordinates of shape (N, 2) [x, y].
        trg_points (torch.Tensor): The target points coordinates of shape (N, 2) [x, y].
        img_size (tuple): The size of the image (height, width).

    Returns:
        torch.Tensor: A batch of boolean masks where True indicates the pixel is within the distance for each point pair.
    """
    src_points = src_points.float()
    trg_points = trg_points.float()

    h, w = img_size
    point_distances = ((src_points - trg_points).norm(dim=1) / (2**0.5)).clamp(min=5)  # Multiplying by 1/sqrt(2)

    y_indices, x_indices = torch.meshgrid(
        torch.arange(h, device=src_points.device),
        torch.arange(w, device=src_points.device),
        indexing="ij"
    )

    # Expand grid to match the batch size (y_indices, x_indices: shape [N, H, W])
    y_indices = y_indices.expand(src_points.size(0), -1, -1)
    x_indices = x_indices.expand(src_points.size(0), -1, -1)

    distance_to_p0 = ((x_indices - src_points[:, None, None, 0])**2 + (y_indices - src_points[:, None, None, 1])**2).sqrt()
    distance_to_p1 = ((x_indices - trg_points[:, None, None, 0])**2 + (y_indices - trg_points[:, None, None, 1])**2).sqrt()
    masks = (distance_to_p0 < point_distances[:, None, None]) | (distance_to_p1 < point_distances[:, None, None]) # (N, H, W)

    return masks

def nn_get_matches(src_featmaps, trg_featmaps, query, l2_norm=True, mask=None):
    '''
    Find the nearest neighbour matches for a given query from source feature maps in target feature maps.

    Args:
        src_featmaps (torch.Tensor): Source feature map with shape (1 x C x H x W).
        trg_featmaps (torch.Tensor): Target feature map with shape (1 x C x H x W).
        query (torch.Tensor): (x, y) coordinates with shape (N x 2), must be in the range of src_featmaps.
        l2_norm (bool): If True, apply L2 normalization to features.
        mask (torch.Tensor): Optional mask with shape (N x H x W).

    Returns:
        torch.Tensor: (x, y) coordinates of the top matches with shape (N x 2).
    '''
    # Extract features from the source feature map at the query points
    _, c, h, w = src_featmaps.shape  # (1, C, H, W)
    query = query.long() 
    src_feat = src_featmaps[0, :, query[:, 1], query[:, 0]]  # (C, N)

    if l2_norm:
        src_feat = F.normalize(src_feat, p=2, dim=0) 
        trg_featmaps = F.normalize(trg_featmaps, p=2, dim=1) 

    trg_featmaps = trg_featmaps.view(c, -1)  # flatten (C, H*W)
    similarity = torch.mm(src_feat.t(), trg_featmaps)  # similarity shape: (N, H*W)

    if mask is not None:
        mask = mask.view(-1, h * w)  # mask shape: (N, H*W)
        similarity = torch.where(mask, similarity, torch.full_like(similarity, -torch.inf))

    # Get the indices of the best matches
    best_match_idx = similarity.argmax(dim=-1)  # best_match_idx shape: (N,)

    # Convert flat indices to 2D coordinates
    y_coords = best_match_idx // w  # y_coords shape: (N,)
    x_coords = best_match_idx % w  # x_coords shape: (N,)
    coords = torch.stack((x_coords, y_coords), dim=1)  # coords shape: (N, 2)

    return coords.float()  # Output shape: (N, 2)

class SDFeaturizer(StableDiffusionPipeline):
    """Used to extract SD2-1 feature for semantic point matching (DIFT)."""
    @torch.no_grad()
    def __call__(
        self,
        img_tensor,
        t=261,
        ensemble=8,
        prompt=None,
        prompt_embeds=None
    ):
        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.mode() * self.vae.config.scaling_factor
        latents = latents.expand(ensemble, -1, -1, -1)
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        if prompt_embeds is None:
            if prompt is None:
                prompt = ""
            prompt_embeds = self.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )[0]
        prompt_embeds = prompt_embeds.expand(ensemble, -1, -1)

        # Cache output of second upblock of unet
        unet_feature = []
        def hook(module, input, output):
            unet_feature.clear()
            unet_feature.append(output)
        handle = list(self.unet.children())[4][1].register_forward_hook(hook=hook) 
        self.unet(latents_noisy, t, prompt_embeds)
        handle.remove()

        return unet_feature[0].mean(dim=0, keepdim=True)

class DragEvaluator:
    def __init__(self):
        self.clip_loaded = False
        self.dino_loaded = False
        self.sd_loaded = False
        self.lpips_loaded = False
    
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float16
    
    def load_dino(self):
        if not self.dino_loaded:
            dino_path = 'facebook/dinov2-base'
            self.dino_model = AutoModel.from_pretrained(dino_path).to(self.device).to(self.dtype)
            self.dino_loaded = True
    
    def load_sd(self):
        if not self.sd_loaded:
            sd_path = 'stabilityai/stable-diffusion-2-1'
            self.sd_feat = SDFeaturizer.from_pretrained(sd_path, torch_dtype=self.dtype).to(self.device)
            self.sd_loaded = True

    def load_lpips(self):
        if not self.lpips_loaded:
            self.loss_fn_alex = lpips.LPIPS(net='alex').to(self.device).to(self.dtype)
            self.lpips_loaded = True

    def preprocess_image(self, image):
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1  # Normalize to [-1, 1]
        image = image.unsqueeze(0).permute(0, 3, 1, 2)  # Rearrange to (1, C, H, W)
        return image.to(self.device).to(self.dtype)
    
    @torch.no_grad()
    def compute_lpips(self, image1, image2):
        """
            Learned Perceptual Image Patch Similarity (LPIPS) metric. (https://richzhang.github.io/PerceptualSimilarity/)
        """
        self.load_lpips()
        
        image1 = self.preprocess_image(image1)
        image2 = self.preprocess_image(image2)
        image1 = F.interpolate(image1, (224,224), mode='bilinear')
        image2 = F.interpolate(image2, (224,224), mode='bilinear')

        return self.loss_fn_alex(image1, image2).item()

    def encode_image(self, image, method, prompt=None):
        if method == 'dino':
            featmap = self.dino_model(image).last_hidden_state[:, 1:, :].permute(0, 2, 1)
            featmap = featmap.view(1, -1, 60, 60) # 60 = 840 / 14
        elif method == 'sd':
            featmap = self.sd_feat(image, prompt=prompt)
        else:
            raise NotImplementedError('Only SD and DINO supported.')
        return featmap
    
    @torch.no_grad()
    def compute_distance(self, src_image, trg_image, src_kps, trg_kps, method, prompt=None, plot_path=None):
        """ Mean Distance Metric """
        if method == 'dino':
            self.load_dino()
        elif method == 'sd':
            self.load_sd()
        else:
            raise NotImplementedError('Only SD and DINO supported.')

        src_kps = torch.tensor(src_kps, device=self.device).to(torch.long) # (N, 2) N points
        trg_kps = torch.tensor(trg_kps, device=self.device).to(torch.long) # (N, 2) N points
        
        # Resize target image and scale target points when necessary
        if src_image.shape != trg_image.shape:
            src_img_h, src_img_w, _ = src_image.shape
            trg_img_h, trg_img_w, _ = trg_image.shape
            trg_image = cv2.resize(trg_image, (src_img_w, src_img_h))

            trg_kps = trg_kps * torch.tensor([src_img_w, src_img_h], device=self.device)
            trg_kps = trg_kps / torch.tensor([trg_img_w, trg_img_h], device=self.device)
            trg_kps = trg_kps.to(torch.long)

        image_h, image_w, _ = src_image.shape        
        image_size = 840 if method == 'dino' else 768

        source_image = self.preprocess_image(src_image) # 1, 3, H, W
        target_image = self.preprocess_image(trg_image) # 1, 3, H, W
        source_image = F.interpolate(source_image, size=(image_size, image_size)) # 1, 3, img_size, img_size
        target_image = F.interpolate(target_image, size=(image_size, image_size)) # 1, 3, img_size, img_size

        src_featmap = self.encode_image(source_image, method=method, prompt=prompt)
        src_featmap = F.interpolate(src_featmap, size=(image_h, image_w))

        trg_featmap = self.encode_image(target_image, method=method, prompt=prompt)
        trg_featmap = F.interpolate(trg_featmap, size=(image_h, image_w))

        mask = create_mask(src_kps, trg_kps, (image_h, image_w))
        pred_trg_kps = nn_get_matches(src_featmap, trg_featmap, src_kps, l2_norm=True, mask=mask) 

        distance = trg_kps - pred_trg_kps
        distance[:, 0] /= image_w; distance[:, 1] /= image_h
        distance = distance.norm(dim=-1).mean().item()

        if plot_path:
            plot_matching_result(src_image, trg_image, src_kps.cpu().numpy(),trg_kps.cpu().numpy(), pred_trg_kps.cpu().numpy(), output_path=plot_path)

        return distance