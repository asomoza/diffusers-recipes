import os

import torch
from diffusers import QwenImagePipeline


device = "cuda"
dtype = torch.bfloat16
repo_id = "Qwen/Qwen-Image-2512"
output_dir = "./outputs/qwen_image_2512"
seed = None
prompt = """A photograph that captures a young woman on a city rooftop, with a hazy city skyline in the background. She has long, dark hair that naturally drapes over her shoulders and is wearing a simple tank top. Her posture is relaxed, with her hands resting on the railing in front of her, leaning slightly forward as she looks directly into the camera. The sunlight, coming from behind her at an angle, creates a soft backlight effect that casts a warm golden halo around the edges of her hair and shoulders. This light also produces a slight lens flare, adding a dreamy quality to the image. The city buildings in the background are blurred by the backlight, emphasizing the main subject. The overall tone is warm, evoking a sense of tranquility and a hint of melancholy."""
negative_prompt = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"

if not seed:
    seed = torch.randint(0, 2**32, (1,)).item()
generator = torch.Generator(device="cpu").manual_seed(seed)

onload_device = torch.device("cuda")
offload_device = torch.device("cpu")

pipe = QwenImagePipeline.from_pretrained(repo_id, torch_dtype=dtype)
pipe.enable_group_offload(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="leaf_level",
    use_stream=True,
    record_stream=True,
)

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=1664,
    height=928,
    num_inference_steps=28,
    true_cfg_scale=4.0,
    generator=generator,
).images[0]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image.save(os.path.join(output_dir, f"pipeline_leaf_stream_group_offload_record_{seed}.png"))
