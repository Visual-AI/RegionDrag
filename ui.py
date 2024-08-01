import torch
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is not available, but required.')

import gradio as gr
from utils.ui_utils import *
from utils.drag import sd_version

LENGTH = 400 # length of image in Gradio App, you can adjust it according to your screen size
GEN_SIZE = {'v1-5': 512, 'v2-1': 768, 'xl': 1024}[sd_version] # Default generated image size 

def main():
    with gr.Blocks() as demo:
        gr_length = gr.Number(value=LENGTH, visible=False, precision=0)
        gr_gen_size = gr.Number(value=GEN_SIZE, visible=False, precision=0)

        selected_masks = gr.State(value=[])
        src_points_m = gr.State(value=None); trg_points_m = gr.State(value=None)
        
        selected_points = gr.State(value=[])
        selected_shapes = gr.State(value=[])
        src_points = gr.State(value=None); trg_points = gr.State(value=None)
        
        selected_points_r = gr.State(value=[])
        src_points_r = gr.State(value=None); trg_points_r = gr.State(value=None)

        seed = gr.Number(value=42, label="Generation Seed", precision=0, visible=False)      
        start_t = gr.Number(value=0.5, visible=False)
        end_t = gr.Number(value=0.2, visible=False)
        
        # layout definition
        with gr.Row():
            gr.Markdown("""
                # Official Implementation of [RegionDrag](https://arxiv.org/abs/2407.18247)  
                #### Explore our detailed [User Guide](https://github.com/LuJingyi-John/RegionDrag/blob/main/README.md) for interface instructions.
            """)

        with gr.Row():
            with gr.Tab(label='Region pairs'):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""<p style="text-align: center; font-size: 20px">1. Upload image and add regions</p>""")
                        canvas_m = gr.Image(type="numpy", tool="sketch", label=" ", height=LENGTH, width=LENGTH)
                        with gr.Row():
                            resize_button_m = gr.Button("Fit Canvas")
                            add_mask_button = gr.Button("Add Region")            
                    with gr.Column():
                        gr.Markdown("""<p style="text-align: center; font-size: 20px">2. View Input</p>""")
                        input_image_m = gr.Image(type="numpy", label=" ", height=LENGTH, width=LENGTH, interactive=False)
                        with gr.Row():
                            undo_mask_button = gr.Button("Undo Region")
                            clear_mask_button = gr.Button("Clear Region")
                    with gr.Column():
                        gr.Markdown("""<p style="text-align: center; font-size: 20px">Results</p>""")
                        output_image_m = gr.Image(type="numpy", label=" ", height=LENGTH, width=LENGTH, interactive=False)
                        with gr.Row():
                            run_button_m = gr.Button("Run Drag")
                            clear_all_button_m = gr.Button("Clear All")
            
            with gr.Tab(label='Polygon pairs'):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""<p style="text-align: center; font-size: 20px">1. Upload image</p>""")
                        canvas = gr.Image(type="numpy", tool="sketch", label=" ", height=LENGTH, width=LENGTH, interactive=True)
                        with gr.Row():
                            resize_button = gr.Button("Fit Canvas")
                    with gr.Column():
                        gr.Markdown("""<p style="text-align: center; font-size: 20px">2. Click vertices</p>""")
                        input_image = gr.Image(type="numpy", label=" ", height=LENGTH, width=LENGTH, interactive=True)
                        with gr.Row():
                            undo_point_button = gr.Button("Undo Point")
                            clear_point_button = gr.Button("Clear Point")
                    with gr.Column():
                        gr.Markdown("""<p style="text-align: center; font-size: 20px">Results</p>""")
                        output_image = gr.Image(type="numpy", label=" ", height=LENGTH, width=LENGTH, interactive=False)
                        with gr.Row():
                            run_button = gr.Button("Run Drag")
                            clear_all_button = gr.Button("Clear All")
                shape = gr.Radio(choices=['▲ Tri', '■ Quad'], value='■ Quad', type='index', label='Mask Shape', interactive=True)
            
            with gr.Tab(label='Region + points'):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""<p style="text-align: center; font-size: 20px">1. Upload image and draw regions</p>""")
                        canvas_r = gr.Image(type="numpy", tool="sketch", label=" ", height=LENGTH, width=LENGTH)
                        with gr.Row():
                            resize_button_r = gr.Button("Fit Canvas")         
                    with gr.Column():
                        gr.Markdown("""<p style="text-align: center; font-size: 20px">2. Click points to control regions</p>""")
                        input_image_r = gr.Image(type="numpy", label=" ", height=LENGTH, width=LENGTH, interactive=True)
                        with gr.Row():
                            undo_point_button_r = gr.Button("Undo Point")
                            clear_point_button_r = gr.Button("Clear Point")
                    with gr.Column():
                        gr.Markdown("""<p style="text-align: center; font-size: 20px">Results</p>""")
                        output_image_r = gr.Image(type="numpy", label=" ", height=LENGTH, width=LENGTH, interactive=False)
                        with gr.Row():
                            run_button_r = gr.Button("Run Drag")
                            clear_all_button_r = gr.Button("Clear All")

        with gr.Tab("Generation Parameters"):
            with gr.Row():
                prompt = gr.Textbox(label="Prompt describing output image (Optional)", value='A photo of an object.')
                data_path = gr.Textbox(value='output/default', label="Output path")
            with gr.Row():
                steps = gr.Slider(minimum=20, maximum=100, value=20, step=20, label='Sampling steps', interactive=True)
                noise_scale = gr.Slider(minimum=0, maximum=1.6, value=0.6, step=0.2, label='Handle Noise Scale', interactive=True) # alpha
                method = gr.Dropdown(choices=['Encode then CP', 'CP then Encode'], value='Encode then CP', label='Method', interactive=True)

        clear_all_button_m.click(
            clear_all_m,
            [gr_length],
            [canvas_m, input_image_m, output_image_m, selected_masks, prompt, data_path, steps, noise_scale, src_points_m, trg_points_m]
        )
        canvas_m.clear(
            clear_all_m,
            [gr_length],
            [canvas_m, input_image_m, output_image_m, selected_masks, prompt, data_path, steps, noise_scale, src_points_m, trg_points_m]
        )
        resize_button_m.click(
            clear_masks,
            [canvas_m, selected_masks],
            [input_image_m]
        ).then(
            resize_image,
            [canvas_m, gr_length, gr_gen_size],
            [canvas_m, input_image_m, output_image_m]
        )
        add_mask_button.click(
            add_mask,
            [canvas_m, selected_masks],
            [input_image_m]
        ).then(
            preview_out_image_m,
            [canvas_m, selected_masks],
            [output_image_m, src_points_m, trg_points_m]
        )
        undo_mask_button.click(
            undo_mask,
            [canvas_m, selected_masks],
            [input_image_m]
        ).then(
            preview_out_image_m,
            [canvas_m, selected_masks],
            [output_image_m, src_points_m, trg_points_m]
        )
        clear_mask_button.click(
            clear_masks,
            [canvas_m, selected_masks],
            [input_image_m]
        ).then(
            preview_out_image_m,
            [canvas_m, selected_masks],
            [output_image_m, src_points_m, trg_points_m]
        )
        run_button_m.click(
            preview_out_image_m,
            [canvas_m, selected_masks],
            [output_image_m, src_points_m, trg_points_m]
        ).then(
            run_process,
            [canvas_m, input_image_m, output_image_m, src_points_m, trg_points_m, prompt, start_t, end_t, steps, noise_scale, data_path, method, seed],
            [output_image_m]
        )

        clear_all_button.click(
            clear_all,
            [gr_length],
            [canvas, input_image, output_image, selected_points, selected_shapes, prompt, data_path, steps, noise_scale, src_points, trg_points]
        )
        canvas.clear(
            clear_all,
            [gr_length],
            [canvas, input_image, output_image, selected_points, selected_shapes, prompt, data_path, steps, noise_scale, src_points, trg_points]
        )
        resize_button.click(
            clear_points,
            [canvas, selected_points, selected_shapes],
            [input_image]
        ).then(
            resize_image,
            [canvas, gr_length, gr_gen_size],
            [canvas, input_image, output_image]
        )
        canvas.edit(
            draw_input_image,
            [canvas, selected_points, selected_shapes],
            input_image
        ).then(
            preview_out_image, 
            [canvas, selected_points, selected_shapes],
            [output_image, src_points, trg_points]
        )
        shape.change(
            update_shape,
            [canvas, shape, selected_points, selected_shapes],
            [input_image]
        ).then(
            preview_out_image, 
            [canvas, selected_points, selected_shapes],
            [output_image, src_points, trg_points]
        )
        input_image.upload(
            wrong_upload,
            outputs=[input_image]
        )
        input_image.select(
            add_point,
            [canvas, shape, selected_points, selected_shapes],
            [input_image]
        ).then(
            preview_out_image, 
            [canvas, selected_points, selected_shapes],
            [output_image, src_points, trg_points]
        )
        undo_point_button.click(
            undo_point,
            [canvas, shape, selected_points, selected_shapes],
            [input_image]
        ).then(
            preview_out_image, 
            [canvas, selected_points, selected_shapes],
            [output_image, src_points, trg_points]
        )
        clear_point_button.click(
            clear_points,
            [canvas, selected_points, selected_shapes],
            [input_image]
        ).then(
            preview_out_image, 
            [canvas, selected_points, selected_shapes],
            [output_image, src_points, trg_points]
        )
        run_button.click(
            preview_out_image, 
            [canvas, selected_points, selected_shapes],
            [output_image, src_points, trg_points]
        ).then(
            run_process, 
            [canvas, input_image, output_image, src_points, trg_points, prompt, start_t, end_t, steps, noise_scale, data_path, method, seed],
            [output_image]
        )

        clear_all_button_r.click(
            clear_all_m,
            [gr_length],
            [canvas_r, input_image_r, output_image_r, selected_points_r, prompt, data_path, steps, noise_scale, src_points_r, trg_points_r]
        )
        canvas_r.clear(
            clear_all_m,
            [gr_length],
            [canvas_r, input_image_r, output_image_r, selected_points_r, prompt, data_path, steps, noise_scale, src_points_r, trg_points_r]
        )
        resize_button_r.click(
            clear_points_r,
            [canvas_r, selected_points_r],
            [input_image_r]
        ).then(
            resize_image,
            [canvas_r, gr_length, gr_gen_size],
            [canvas_r, input_image_r, output_image_r]
        )
        canvas_r.edit(
            draw_input_image_r,
            [canvas_r, selected_points_r],
            [input_image_r]
        ).then(
            preview_out_image_r, 
            [canvas_r, selected_points_r],
            [output_image_r, src_points_r, trg_points_r]
        )
        input_image_r.upload(
            wrong_upload,
            outputs=[input_image_r]
        )
        input_image_r.select(
            add_point_r,
            [canvas_r, selected_points_r],
            [input_image_r]
        ).then(
            preview_out_image_r, 
            [canvas_r, selected_points_r],
            [output_image_r, src_points_r, trg_points_r]
        )
        undo_point_button_r.click(
            undo_point_r,
            [canvas_r, selected_points_r],
            [input_image_r]
        ).then(
            preview_out_image_r, 
            [canvas_r, selected_points_r],
            [output_image_r, src_points_r, trg_points_r]
        )
        clear_point_button_r.click(
            clear_points_r,
            [canvas_r, selected_points_r],
            [input_image_r]
        ).then(
            preview_out_image_r, 
            [canvas_r, selected_points_r],
            [output_image_r, src_points_r, trg_points_r]
        )
        run_button_r.click(
            preview_out_image_r, 
            [canvas_r, selected_points_r],
            [output_image_r, src_points_r, trg_points_r]
        ).then(
            run_process,
            [canvas_r, input_image_r, output_image_r, src_points_r, trg_points_r, prompt, start_t, end_t, steps, noise_scale, data_path, method, seed],
            [output_image_r]
        )

    demo.queue().launch(share=True, debug=True)

if __name__ == '__main__':
    main()