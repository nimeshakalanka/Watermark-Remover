import gradio as gr
import torch
from PIL import Image
import numpy as np

# --- CONFIGURATION ---
MODEL_PATH = "big-lama.pt"

print("Loading AI Model from local file...")
try:
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(f"Make sure '{MODEL_PATH}' is in the same folder as this script!")
    exit()


def process_image(input_dict):
    if input_dict is None:
        return None

    image_pil = input_dict["background"].convert("RGB")
    mask_pil = input_dict["layers"][0].split()[-1].convert("L")
    w, h = image_pil.size

    new_w = (w // 8) * 8
    new_h = (h // 8) * 8

    img_resized = image_pil.resize((new_w, new_h))
    mask_resized = mask_pil.resize((new_w, new_h))

    img_tensor = torch.from_numpy(np.array(img_resized)).float().div(255.0).permute(2, 0, 1).unsqueeze(0)
    mask_tensor = torch.from_numpy(np.array(mask_resized)).float().div(255.0).unsqueeze(0).unsqueeze(0)

    mask_tensor = (mask_tensor > 0).float()

    with torch.no_grad():
        try:
            result_tensor = model(img_tensor, mask_tensor)
        except:
            return image_pil

    result_tensor = result_tensor[0].permute(1, 2, 0).clamp(0, 1) * 255
    result_np = result_tensor.byte().cpu().numpy()
    result_image = Image.fromarray(result_np)

    return result_image


css = """
.container { max-width: 1100px; margin: auto; padding-top: 20px; }
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💧 Manual Watermark Remover")
    gr.Markdown(f"Using local model: `{MODEL_PATH}`")

    with gr.Row():
        input_img = gr.ImageEditor(
            label="Draw over Watermark",
            type="pil",
            brush=gr.Brush(colors=["#FFFFFF"], default_size=20),
            interactive=True,
            height=500
        )
        output_img = gr.Image(label="Result", height=500)

    btn = gr.Button("Remove Watermark", variant="primary")
    btn.click(process_image, inputs=input_img, outputs=output_img)

if __name__ == "__main__":
    demo.launch(inbrowser=True)