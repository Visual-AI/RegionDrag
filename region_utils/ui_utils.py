import os
import pickle
import cv2
import numpy as np
from PIL import Image
import gradio as gr
import torch
import time
import functools

from .drag import drag

# 1. Common utils
def timeit(func):
    """Decorator to measure the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  
        result = func(*args, **kwargs)
        end_time = time.time()  
        elapsed_time = end_time - start_time  
        print(f"Function {func.__name__!r} took {elapsed_time:.4f} seconds to complete.")
        return result

    return wrapper

def get_W_H(max_length, aspect_ratio):
    height = int(max_length / aspect_ratio) if aspect_ratio >= 1 else max_length
    width = max_length if aspect_ratio >= 1 else int(max_length * aspect_ratio)
    height = int(height / 8) * 8
    width = int(width / 8) * 8
    return width, height

def canvas_to_image_and_mask(canvas):
    """Extracts the image (H, W, 3) and the mask (H, W) from a Gradio canvas object."""
    image = canvas["image"].copy()
    mask = np.uint8(canvas["mask"][:, :, 0] > 0).copy()
    return image, mask

def draw_mask_border(image, mask):
    """Find the contours of shapes in the mask and draw them on the image."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255, 255, 255), 2)

def mask_image(image, mask, color=[255,0,0], alpha=0.3):
    """Apply a binary mask to an image, highlighting the masked regions with a specified color and transparency."""
    out = image.copy()
    out[mask == 1] = color
    out = cv2.addWeighted(out, alpha, image, 1-alpha, 0, out)
    return out

def resize_image(canvas, canvas_length, image_length):
    """Fit the image to an appropriate size."""
    if canvas is None:
        return (gr.Image(value=None, width=canvas_length, height=canvas_length),) * 3

    image = canvas_to_image_and_mask(canvas)[0]
    image_h0, image_w0, _ = image.shape
    image_w1, image_h1 = get_W_H(image_length, image_w0 / image_h0)
    image = cv2.resize(image, (image_w1, image_h1))

    # helpful when uploaded gradio image having width > length
    canvas_w1, canvas_h1 = (canvas_length, canvas_length) if image_h0 > image_w0 else get_W_H(canvas_length, image_w0 / image_h0)
    return (gr.Image(value=image, width=canvas_w1, height=canvas_h1),) * 3

def wrong_upload():
    """Prevent user to upload an image on the second column"""
    gr.Warning('You should upload an image on the left.')
    return None

def save_data(original_image, input_image, preview_image, mask, src_points, trg_points, prompt, data_path):
    """
        save following under `data_path` directory
        (1) original.png        : original image
        (2) input_image.png     : input image
        (3) preview_image.png   : preview image
        (4) meta_data_mask.pkl  : {'prompt' : text prompt, 
                                   'points' : [(x1,y1), (x2,y2), ..., (xn, yn)],
                                   'mask'   : binary mask np.uint8 (H, W)}
        (x1, y1), (x3, y3), ... are from source (handle) points
        (x2, y2), (x4, y4), ... are from target points
    """      
    os.makedirs(data_path, exist_ok=True)
    img_path = os.path.join(data_path, 'original_image.png')
    input_img_path = os.path.join(data_path, 'user_drag_region.png')
    preview_img_path = os.path.join(data_path, 'preview.png')
    meta_data_path = os.path.join(data_path, 'meta_data_mask.pkl')
    
    Image.fromarray(original_image).save(img_path)
    Image.fromarray(input_image).save(input_img_path)
    Image.fromarray(preview_image).save(preview_img_path)

    points = [point for pair in zip(src_points, trg_points) for point in pair]

    meta_data = {
        'prompt': prompt,
        'points': points,
        'mask': mask
    }
    with open(meta_data_path, 'wb') as file:
        pickle.dump(meta_data, file, protocol=pickle.HIGHEST_PROTOCOL)

@torch.no_grad()
def run_process(canvas, input_image, preview_image, src_points, trg_points, prompt, start_t, end_t, steps, noise_scale, data_path, method, seed, progress=gr.Progress()):
    if canvas is None:
        return None
    
    original_image, mask = canvas_to_image_and_mask(canvas)
    mask = np.ones_like(mask)
    
    if src_points is None or len(src_points) == 0:
        return original_image
    
    drag_data = {
        'ori_image' : original_image, 
        'preview' : preview_image, 
        'prompt' : prompt, 
        'mask' : mask, 
        'source' : src_points, 
        'target' : trg_points
    }

    return drag(drag_data, steps, start_t, end_t, noise_scale, seed, progress.tqdm, method, data_path)

# 2. mask utils (region pairs)
def clear_all_m(length):
    """Used to initialize all inputs in ui's region pair tab."""
    return (gr.Image(value=None, height=length, width=length),) * 3 + \
        ([], "A photo of an object.", "output/default", 20, 0.6, None, None)

def draw_input_image_m(canvas, selected_masks):
    """Draw an image reflecting user's intentions."""
    image = canvas_to_image_and_mask(canvas)[0]
    for i, mask in enumerate(selected_masks):
        color = [255, 0, 0] if i % 2 == 0 else [0, 0, 255]
        image = mask_image(image, mask, color=color, alpha=0.3)
        draw_mask_border(image, mask)
    return image

@torch.no_grad()
def region_pair_to_pts(src_region, trg_region, scale=1):
    """
        Perform dense mapping beween one source (handle) and one target region.
        `scale` is set to 1/8 for mapping in SD latent space.
    """

    def mask_min_max(tensor, mask, dim=None):
        """
            Compute the masked max or min of a tensor along a given dimension.
        """
        # Apply the mask by using a very small or very large number for min/max respectively
        masked_tensor = torch.where(mask, tensor, torch.inf)
        masked_min = torch.min(masked_tensor, dim=dim)[0] if dim is not None else torch.min(masked_tensor)

        masked_tensor = torch.where(mask, tensor, -torch.inf)
        masked_max = torch.max(masked_tensor, dim=dim)[0] if dim is not None else torch.max(masked_tensor)
        return masked_min, masked_max
    
    src_region = cv2.resize(src_region, (int(src_region.shape[1]*scale), int(src_region.shape[0]*scale)))
    trg_region = cv2.resize(trg_region, (int(trg_region.shape[1]*scale), int(trg_region.shape[0]*scale)))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    src_region = torch.from_numpy(src_region).to(device).bool()
    trg_region = torch.from_numpy(trg_region).to(device).bool()

    h, w = src_region.shape
    src_grid = trg_grid = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij'), dim=-1).to(device).float()
    trg_pts = trg_grid[torch.where(trg_region)]

    src_x_min, src_x_max = mask_min_max(src_grid[:, :, 1], mask=src_region)
    trg_x_min, trg_x_max = mask_min_max(trg_grid[:, :, 1], mask=trg_region)

    scale_x = (src_x_max - src_x_min) / (trg_x_max - trg_x_min).clamp(min=1e-4)

    trg_grid[:, :, 1] = ((trg_grid[:, :, 1] - trg_x_min) * scale_x + src_x_min)
    trg_grid[:, :, 1] = torch.where(trg_region, trg_grid[:, :, 1], 0)

    src_y_min, src_y_max = mask_min_max(src_grid[:, :, 0], mask=src_region, dim=0)
    trg_y_min, trg_y_max = mask_min_max(trg_grid[:, :, 0], mask=trg_region, dim=0)
    src_y_min, src_y_max = src_y_min[trg_grid[:, :, 1].int()], src_y_max[trg_grid[:, :, 1].int()]

    scale_y = (src_y_max - src_y_min) / (trg_y_max - trg_y_min).clamp(min=1e-4)
    trg_grid[:, :, 0] = ((trg_grid[:, :, 0] - trg_y_min) * scale_y + src_y_min)
    warp_trg_pts = trg_grid[torch.where(trg_region)]

    return warp_trg_pts[:, [1, 0]].int(), trg_pts[:, [1, 0]].int()

def preview_out_image_m(canvas, selected_masks):
    """Preview the output image by directly copy-pasting pixel values."""
    if canvas is None:
        return None, None, None
    image = canvas_to_image_and_mask(canvas)[0]

    if len(selected_masks) < 2:
        return image, None, None
    
    src_regions = selected_masks[0:-1:2]
    trg_regions = selected_masks[1::2]
    
    src_points, trg_points = map(torch.cat, zip(*[region_pair_to_pts(src_region, trg_region) for src_region, trg_region in zip(src_regions, trg_regions)]))
    src_idx, trg_idx = src_points[:, [1, 0]].cpu().numpy(), trg_points[:, [1, 0]].cpu().numpy()
    image[tuple(trg_idx.T)] = image[tuple(src_idx.T)]        

    src_points, trg_points = map(torch.cat, zip(*[region_pair_to_pts(src_region, trg_region, scale=1/8) for src_region, trg_region in zip(src_regions, trg_regions)]))
    return image, src_points*8, trg_points*8

def add_mask(canvas, selected_masks):
    """Add a drawn mask, and draw input image"""
    if canvas is None:
        return None 
    image, mask = canvas_to_image_and_mask(canvas)
    if len(selected_masks) >= 1 and (mask == 0).all():
        gr.Warning('Do not input empty region.')
    else:
        selected_masks.append(mask)
    return draw_input_image_m(canvas, selected_masks)

def undo_mask(canvas, selected_masks):
    """Undo a drawn mask, and draw input image"""
    if len(selected_masks) > 0:
        selected_masks.pop()
    if canvas is None:
        return None 
    return draw_input_image_m(canvas, selected_masks)

def clear_masks(canvas, selected_masks):
    """Clear all drawn masks, and draw input image"""
    selected_masks.clear()
    if canvas is None:
        return None 
    return draw_input_image_m(canvas, selected_masks)

# 3. vertice utils (polygon pairs)
def clear_all(length):
    """Used to initialize all inputs in ui's vertice pair tab."""
    return (gr.Image(value=None, height=length, width=length),) * 3 + \
        ([], [], "A photo of an object.", "output/default", 20, 0.6, None, None)

def draw_input_image(canvas, selected_points, selected_shapes):
    """Draw input image with vertices."""
    # Extract the image and mask from the canvas
    image, mask = canvas_to_image_and_mask(canvas)
    if mask.sum() > 0:
        gr.Info('The drawn mask is not used.')
    mask = np.ones_like(mask)
    # If a mask is present (i.e., sum of mask values > 0), non-masked parts will be darkened
    masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3) if mask.sum() > 0 else image

    def draw_circle(image, point, text, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        font_color = (255, 255, 255)  
        font_thickness = 1
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        bottom_left_corner = (point[0] - text_size[0] // 2, point[1] + text_size[1] // 2)
        
        cv2.circle(image, tuple(point), 8, color, -1)
        cv2.circle(image, tuple(point), 8, [255, 255, 255])
        cv2.putText(image, text, bottom_left_corner, font, font_scale, font_color, font_thickness)

    def draw_polygon(image, points, shape, color):
        if len(points) != shape:
            return image 
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        points = np.array(points).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points], 1)
        return mask_image(image, mask, color, alpha=0.3)
    
    if len(selected_points) == 0:
        return masked_img
    
    start_idx = 1
    for points, shape in zip(selected_points, selected_shapes):
        src_pts, trg_pts = points[:shape], points[shape:]
        masked_img = draw_polygon(masked_img, src_pts, shape, color=[255, 0, 0])
        masked_img = draw_polygon(masked_img, trg_pts, shape, color=[0, 0, 255])

        for i, src_pt in enumerate(src_pts, start=start_idx):
            draw_circle(masked_img, src_pt, str(i), [255, 102, 102])
        for i, trg_pt in enumerate(trg_pts, start=start_idx):
            draw_circle(masked_img, trg_pt, str(i), [102, 102, 255])
        start_idx = i + 1

    return masked_img

def transform_polygon(src_points, trg_points, scale=1):
    """
        Perform dense mapping with source (handle) and target triangle or quadrilateral.
    """
    def get_points_inside_polygon(points):
        points = np.array(points, dtype=np.int32)
        x_max, y_max = np.amax(points, axis=0) + 1
        mask = np.zeros((y_max, x_max), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 1)
        return np.column_stack(np.where(mask == 1))[:, [1, 0]]

    if len(trg_points) not in [3, 4]:
        raise NotImplementedError('Only triangles and quadrilaterals are implemented')

    src_points, trg_points = np.float32(src_points)*scale, np.float32(trg_points)*scale
    M = (cv2.getAffineTransform if len(trg_points) == 3 else cv2.getPerspectiveTransform)(trg_points, src_points)
    points_inside = get_points_inside_polygon(trg_points)
    warped_points = (cv2.transform if len(trg_points) == 3 else cv2.perspectiveTransform)(np.array([points_inside], dtype=np.float32), M)

    return warped_points[0].astype(np.int32), points_inside

def preview_out_image(canvas, selected_points, selected_shapes):
    if canvas is None:
        return None, None, None
    image = canvas_to_image_and_mask(canvas)[0]

    selected_points = selected_points.copy()
    selected_shapes = selected_shapes.copy()

    if len(selected_points) == 0:
        return image, None, None

    if len(selected_points[-1]) != selected_shapes[-1] * 2:
        selected_points.pop()
        selected_shapes.pop()
        if len(selected_points) == 0:
            return image, None, None

    src_points, trg_points = map(np.concatenate, zip(*[transform_polygon(sp[:ss], sp[ss:]) for sp, ss in zip(selected_points, selected_shapes)]))
    src_idx, trg_idx = src_points[:, [1, 0]], trg_points[:, [1, 0]]
    image[tuple(trg_idx.T)] = image[tuple(src_idx.T)]

    src_points, trg_points = map(np.concatenate, zip(*[transform_polygon(sp[:ss], sp[ss:], scale=1/8) for sp, ss in zip(selected_points, selected_shapes)]))
    return image, src_points*8, trg_points*8

def add_point(canvas, shape, selected_points, selected_shapes, evt: gr.SelectData):
    """Collect the selected point, and draw the input image."""
    if canvas is None:
        return None
    
    def is_valid_quadrilateral(p1, p2, p3, p4):
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0  # collinear
            return 1 if val > 0 else 2  # clock or counterclock wise

        def do_intersect(a1, a2, b1, b2):
            o1 = orientation(a1, a2, b1)
            o2 = orientation(a1, a2, b2)
            o3 = orientation(b1, b2, a1)
            o4 = orientation(b1, b2, a2)
            return (o1 != o2 and o3 != o4) and not (o1 == 0 or o2 == 0 or o3 == 0 or o4 == 0)

        # Check for collinearity and intersection between non-adjacent edges
        return not (orientation(p1, p2, p3) == 0 or
                    orientation(p1, p2, p4) == 0 or
                    orientation(p2, p3, p4) == 0 or
                    orientation(p1, p3, p4) == 0 or
                    do_intersect(p1, p2, p3, p4) or
                    do_intersect(p2, p3, p1, p4))

    if len(selected_points) == 0 or len(selected_points[-1]) == 2 * selected_shapes[-1]:
        selected_points.append([evt.index])
        selected_shapes.append(shape+3)
    else:
        selected_points[-1].append(evt.index)
        if selected_shapes[-1] == 4 and len(selected_points[-1]) % selected_shapes[-1] == 0:
            if not is_valid_quadrilateral(*selected_points[-1][-4:]):
                gr.Warning('The drawn quadrilateral is not valid.')
                selected_points[-1].pop()

    return draw_input_image(canvas, selected_points, selected_shapes)

def update_shape(canvas, shape, selected_points, selected_shapes):
    """Allow users to switch between different shape options"""
    if canvas is None:
        return None
    if len(selected_points) == 0:
        return draw_input_image(canvas, selected_points, selected_shapes)
    if len(selected_points[-1]) == selected_shapes[-1] * 2:
        return draw_input_image(canvas, selected_points, selected_shapes)
        
    selected_shapes[-1] = shape + 3
    selected_points[-1] = selected_points[-1][ : (shape+3)*2]
    return draw_input_image(canvas, selected_points, selected_shapes)

def undo_point(canvas, shape, selected_points, selected_shapes):
    """Remove the last added point, and draw the input image."""
    if canvas is None:
        return None 
    if len(selected_points) == 0:
        return draw_input_image(canvas, selected_points, selected_shapes)

    selected_points[-1].pop()
    if len(selected_points[-1]) == 0:
        selected_points.pop()
        selected_shapes.pop()
    update_shape(canvas, shape, selected_points, selected_shapes)
    return draw_input_image(canvas, selected_points, selected_shapes)

def clear_points(canvas, selected_points, selected_shapes):
    """Clear all the points"""
    selected_points.clear()
    selected_shapes.clear()
    if canvas is None:
        return None 
    return draw_input_image(canvas, selected_points, selected_shapes)
    
# 4. region utils (region + point pair)
def draw_input_image_r(canvas, selected_points):
    image, mask = canvas_to_image_and_mask(canvas)
    image = mask_image(image, mask, color=[255, 0, 0], alpha=0.3)
    draw_mask_border(image, mask)
    for idx, point in enumerate(selected_points, start=1):
        if idx % 2 == 0:
            cv2.circle(image, tuple(point), 10, (0, 0, 255), -1)
            cv2.arrowedLine(image, last_point, point, (255, 255, 255), 4, tipLength=0.5)
        else:
            cv2.circle(image, tuple(point), 10, (255, 0, 0), -1)
            last_point = point
    return image


def region_to_points(region_mask, selected_points, scale=1):
    """
    Process a region mask and a list of selected points to find corresponding
    source and target points within the region scaled by a factor.
    """
    def resize_region_mask(mask, scale_factor):
        """Resize the region mask by a scale factor."""
        H, W = mask.shape
        new_H, new_W = (int(H * scale_factor), int(W * scale_factor))
        return cv2.resize(mask, (new_W, new_H)), (new_H, new_W)
    
    def find_contours(mask):
        """Find contours in the mask."""
        return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    def scale_points(points, scale_factor):
        """Scale points by a scale factor."""
        return (np.array(points) * scale_factor).astype(np.int32)

    def draw_mask_from_contours(contour, shape):
        """Draws a mask from given contours."""
        mask = np.zeros(shape, dtype=np.uint8)
        contour = contour[:, np.newaxis, :] if contour.ndim == 2 else contour
        cv2.drawContours(mask, [contour], -1, color=255, thickness=cv2.FILLED)
        return mask

    def find_points_in_mask(mask):
        """Find the coordinates of non-zero points in the mask."""
        return np.column_stack(np.where(mask)).astype(np.int32)[:, [1, 0]]

    def filter_points_by_bounds(source_points, target_points, bounds):
        """Filter points by checking if they fall within the given bounds."""
        height, width = bounds
        src_condition = (
            (source_points[:, 0] >= 0) & (source_points[:, 0] < width) &
            (source_points[:, 1] >= 0) & (source_points[:, 1] < height)
        )
        trg_condition = (
            (target_points[:, 0] >= 0) & (target_points[:, 0] < width) &
            (target_points[:, 1] >= 0) & (target_points[:, 1] < height)
        )
        condition = src_condition & trg_condition
        return source_points[condition], target_points[condition]

    def find_matching_points(source_region, source_points, target_points):
        """Find source points in source_region and their matching target points."""
        match_indices = np.all(source_points[:, None] == source_region, axis=2).any(axis=1)
        return source_points[match_indices], target_points[match_indices]

    def interpolate_points(region_points, reference_points, directions, max_num=100):
        """Interpolate points within a region based on reference points and their directions."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert input numpy arrays to PyTorch tensors and send them to the appropriate device
        region_points = torch.from_numpy(region_points).half().to(device)
        reference_points = torch.from_numpy(reference_points).half().to(device)
        directions = torch.from_numpy(directions).half().to(device)
        
        if len(reference_points) < 2:
            return (region_points + directions).int().to('cpu').numpy()
        
        if len(reference_points) > max_num:
            indices = torch.linspace(0, len(reference_points) - 1, steps=max_num).long().to(device)
            reference_points = reference_points[indices]
            directions = directions[indices]

        distance = torch.norm(region_points.unsqueeze(1) - reference_points.unsqueeze(0), dim=-1)
        _, indices = torch.sort(distance, dim=1)
        indices = indices[:, :min(4, reference_points.shape[0])]

        directions = torch.gather(directions.unsqueeze(0).expand(region_points.size(0), -1, -1), 1, indices.unsqueeze(-1).expand(-1, -1, directions.size(-1)))

        inv_distance = 1 / (torch.gather(distance, 1, indices) + 1e-4)
        weight = inv_distance / inv_distance.sum(dim=1, keepdim=True)

        estimated_direction = (weight.unsqueeze(-1) * directions).sum(dim=1)
        return (region_points + estimated_direction).round().int().to('cpu').numpy()
    
    resized_mask, new_size = resize_region_mask(region_mask, scale)

    contours = find_contours(resized_mask)
    source_points = scale_points(selected_points[0:-1:2], scale)
    target_points = scale_points(selected_points[1::2], scale)

    source_regions = [np.zeros((0, 2)),]; target_regions = [np.zeros((0, 2)),]
    for contour in contours:
        # find point pairs used to manipulate the region inside this contour
        source_contour = contour[:, 0, :]
        source_region_points = find_points_in_mask(draw_mask_from_contours(source_contour, new_size))
        source, target = find_matching_points(source_region_points, source_points, target_points)

        # interplote to find motion of contour points and points inside
        if len(source) == 0:
            continue
        target_contour = interpolate_points(source_contour, source, target - source)
        interpolated_target_points = interpolate_points(source_region_points, source, target - source)
        
        # similar to above, this step ensures that we can have a reference point for each point inside the target region
        target_region_points = find_points_in_mask(draw_mask_from_contours(target_contour, new_size))
        interpolated_source_points = interpolate_points(target_region_points, interpolated_target_points, source_region_points - interpolated_target_points)

        filtered_source, filtered_target = filter_points_by_bounds(interpolated_source_points, target_region_points, new_size)
        source_regions.append(filtered_source)
        target_regions.append(filtered_target)

    return np.concatenate(source_regions).astype(np.int32), np.concatenate(target_regions).astype(np.int32)
    
def preview_out_image_r(canvas, selected_points):
    if canvas is None:
        return None
    image, region_mask = canvas_to_image_and_mask(canvas)

    if len(selected_points) < 2:
        return image, None, None

    src_points, trg_points = region_to_points(region_mask, selected_points)
    image[trg_points[:, 1], trg_points[:, 0]] = image[src_points[:, 1], src_points[:, 0]]
    src_points, trg_points = region_to_points(region_mask, selected_points, scale=1/8)
    return image, src_points*8, trg_points*8

def add_point_r(canvas, selected_points, evt: gr.SelectData):
    if canvas is None:
        return None 
    selected_points.append(evt.index)
    return draw_input_image_r(canvas, selected_points)

def undo_point_r(canvas, selected_points):
    if canvas is None:
        return None 
    if len(selected_points) > 0:
        selected_points.pop()
    return draw_input_image_r(canvas, selected_points)

def clear_points_r(canvas, selected_points):
    if canvas is None:
        return None 
    selected_points.clear()
    return draw_input_image_r(canvas, selected_points)
