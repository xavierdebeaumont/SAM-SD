import gradio as gr 
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

print("PyTorch version:", torch.__version__)
print("CUDA is available:", torch.cuda.is_available())

device = "cuda"
sam_checkpoint = "weights/sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)

pipe = pipe.to(device)

selected_pixels = []

with gr.Blocks() as demo:
    with gr.Row():
        input_img = gr.Image(label="Input")
        mask_img = gr.Image(label="Mask")
        output_img = gr.Image(label="Output")

    with gr.Row():
        prompt_text = gr.Textbox(lines=1, label="Prompt")

    with gr.Row():
        submit = gr.Button("Submit")

    def generate_mask(image, evt:gr.SelectData):
        selected_pixels.append(evt.index)

        predictor.set_image(image)
        input_points = np.array(selected_pixels)
        input_labels = np.ones(input_points.shape[0])
        mask, _, _= predictor.predict(point_coords=input_points,point_labels=input_labels, multimask_output=False)

        mask = Image.fromarray(mask[0, :, :])
        return mask
    
    def inpaint(image, mask, prompt):
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        image = image.resize((512,512))
        mask = mask.resize((512,512))

        output = pipe(prompt=prompt, image=image, mask_image=mask,).images[0]

        return output

    input_img.select(generate_mask, [input_img], [mask_img])
    submit.click(inpaint, inputs=[input_img, mask_img, prompt_text], outputs=[output_img],)

    if __name__ =="__main__":
        demo.launch()
