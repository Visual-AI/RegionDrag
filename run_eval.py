import os
import argparse
import torch
from tqdm import tqdm
import gradio as gr

from region_utils.drag import drag, get_drag_data, get_meta_data
from region_utils.evaluator import DragEvaluator

# Setting up the argument parser
parser = argparse.ArgumentParser(description='Run the drag operation.')
parser.add_argument('--data_dir', type=str, default='drag_data/dragbench-dr/') # OR 'drag_data/dragbench-sr/'
args = parser.parse_args()

evaluator = DragEvaluator()
all_distances = []; all_lpips = []

data_dir = args.data_dir
data_dirs = [dirpath for dirpath, dirnames, _ in os.walk(data_dir) if not dirnames]

start_t = 0.5
end_t = 0.2
steps = 20
noise_scale = 1.
seed = 42

for data_path in tqdm(data_dirs):
    # Region-based Inputs for Editing
    drag_data = get_drag_data(data_path)
    ori_image = drag_data['ori_image']
    out_image = drag(drag_data, steps, start_t, end_t, noise_scale, seed, progress=gr.Progress())

    # Point-based Inputs for Evaluation
    meta_data_path = os.path.join(data_path, 'meta_data.pkl')
    prompt, _, source, target = get_meta_data(meta_data_path)    

    all_distances.append(evaluator.compute_distance(ori_image, out_image, source, target, method='sd', prompt=prompt))
    all_lpips.append(evaluator.compute_lpips(ori_image, out_image))

if all_distances:
    mean_dist = torch.tensor(all_distances).mean().item()
    mean_lpips = torch.tensor(all_lpips).mean().item()
    print(f'MD: {mean_dist:.4f}\nLPIPS: {mean_lpips:.4f}\n')