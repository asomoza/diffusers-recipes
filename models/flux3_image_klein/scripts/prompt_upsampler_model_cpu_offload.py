import os

import torch
from diffusers import ComponentsManager, ModularPipeline


device = "cuda"
dtype = torch.bfloat16
repo_id = "black-forest-labs/flux-3-image-klein"
output_dir = "./outputs/flux3_image_klein"
seed = 7
prompt = """A small neighborhood pet shop: a tabby cat on the counter, a golden retriever in a wicker basket and a scarlet macaw perched above an aquarium. The sign over the door reads WHISKERS & WAGS."""

if not seed:
    seed = torch.randint(0, 2**32, (1,)).item()
generator = torch.Generator(device="cpu").manual_seed(seed)

manager = ComponentsManager()
pipe = ModularPipeline.from_pretrained(repo_id, components_manager=manager)
pipe.load_components(dtype=dtype)
manager.enable_auto_cpu_offload(device=device)

output = pipe(
    prompt=prompt,
    height=1152,
    width=2048,
    generator=generator,
    output=["images", "prompt"],
)
print(output["prompt"])

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output["images"][0].save(os.path.join(output_dir, f"prompt_upsampler_model_cpu_offload_{seed}.png"))
