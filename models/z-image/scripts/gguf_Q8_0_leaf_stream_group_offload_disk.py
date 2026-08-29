import os

import torch
from diffusers import GGUFQuantizationConfig, ZImagePipeline, ZImageTransformer2DModel


device = "cuda"
dtype = torch.bfloat16
repo_id = "Tongyi-MAI/Z-Image-Turbo"
output_dir = "./outputs/zimage"
seed = None
prompt = """A classroom setting with a large green chalkboard on a wooden frame, illuminated by soft morning light from a window on the left. On the chalkboard, written in clear white chalk handwriting: 'Welcome to Diffusers', 'the library that empowers you to create, customize, and experiment with state of the art diffusion models.' Additional chalk notes appear underneath, illustrating what's possible like 'text-to-image', 'image-to-image', 'inpainting', and 'fine-tuning' sketched alongside tiny doodles of gears, sparkles, and miniature image frames. Wooden desks and chairs fill the room, each desk containing a natural scatter of notebooks, textbooks, and pencils. A couple of open books reveal diagrams of AI model architecture and diffusion processes. The walls are decorated with educational posters—some depicting mathematical formulas. Color scheme: muted greens, warm browns, and crisp white. Shallow depth of field emphasizes the chalkboard."""

if not seed:
    seed = torch.randint(0, 2**32, (1,)).item()
generator = torch.Generator(device="cpu").manual_seed(seed)

onload_device = torch.device("cuda")
offload_device = torch.device("cpu")

transformer = ZImageTransformer2DModel.from_single_file(
    "https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/blob/main/z_image_turbo-Q8_0.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)

pipe = ZImagePipeline.from_pretrained(
    repo_id,
    transformer=transformer,
    torch_dtype=dtype,
)

pipe.enable_group_offload(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="leaf_level",
    use_stream=True,
    record_stream=True,
    offload_to_disk_path="./offload_temp",
)

image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=generator,
).images[0]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image.save(os.path.join(output_dir, f"gguf_Q8_0_leaf_stream_group_offload_disk_{seed}.png"))
