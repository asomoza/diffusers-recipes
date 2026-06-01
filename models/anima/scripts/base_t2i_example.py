import os

import torch
from diffusers import ModularPipeline


device = "cuda"
dtype = torch.bfloat16
repo_id = "circlestone-labs/Anima-Base-v1.0-Diffusers"
output_dir = "./outputs/anima"
seed = None
prompt = """masterpiece, best quality, score_7, A colossal humanoid robot over one hundred meters tall stands motionless in the center of a dense futuristic city, dwarfing the surrounding skyscrapers. Its body is an intricate assembly of armored plates, exposed mechanical joints, hydraulic pistons, cooling vents, and thousands of interconnected components that suggest immense engineering complexity. The metal surface shows years of wear, with scratches, scuffs, burn marks, and subtle weathering that add realism and history. A powerful energy reactor glows faintly within its chest, casting soft light through narrow seams in the armor, while its eyes shine like distant beacons high above the streets.
At ground level, the asphalt beneath its feet has fractured under its enormous weight, sending cracks through intersections and sidewalks. Abandoned vehicles, overturned buses, and scattered debris line the street, emphasizing the machine's overwhelming scale. Dust and fine particles drift through the air around its legs, illuminated by shafts of sunlight filtering between towering buildings. Glass skyscrapers reflect the giant machine from every angle, creating a striking interplay of light, metal, and architecture. Tiny human figures gather in the distance, barely visible against the vast urban landscape, looking up in awe.
The scene is viewed from a dramatic low angle near street level, making the robot appear even more imposing as it rises into the sky. Thick clouds part above the city, allowing warm sunlight to stream through the urban canyon, creating long shadows, atmospheric haze, and a sense of grandeur. Every surface is rendered with exceptional detail, from the intricate machinery inside the robot's joints to the reflections in the surrounding windows, resulting in an epic cinematic image that conveys both technological wonder and overwhelming scale."""
negative_prompt = "worst quality, low quality, score_1, score_2, score_3, artist name, letterboxed"

if not seed:
    seed = torch.randint(0, 2**32, (1,)).item()
generator = torch.Generator(device="cpu").manual_seed(seed)

pipe = ModularPipeline.from_pretrained(repo_id)
pipe.load_components(torch_dtype=dtype)
pipe.to(device)

image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=30, generator=generator).images[0]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image.save(os.path.join(output_dir, f"base_t2i_example_{seed}.png"))
