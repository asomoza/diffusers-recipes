import os

import torch
from diffusers import QwenImagePipeline


pipe = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image-2512",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

prompt = """A photograph that captures a young woman on a city rooftop, with a hazy city skyline in the background. She has long, dark hair that naturally drapes over her shoulders and is wearing a simple tank top. Her posture is relaxed, with her hands resting on the railing in front of her, leaning slightly forward as she looks directly into the camera. The sunlight, coming from behind her at an angle, creates a soft backlight effect that casts a warm golden halo around the edges of her hair and shoulders. This light also produces a slight lens flare, adding a dreamy quality to the image. The city buildings in the background are blurred by the backlight, emphasizing the main subject. The overall tone is warm, evoking a sense of tranquility and a hint of melancholy."""
negative_prompt = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=1664,
    height=928,
    num_inference_steps=28,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]

if not os.path.exists("./outputs/qwen-image"):
    os.makedirs("./outputs/qwen-image")

image.save("./outputs/qwen-image/base_example.png")
