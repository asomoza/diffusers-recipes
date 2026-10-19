import os

import torch
from diffusers import ComponentsManager, ModularPipeline
from sdnq import SDNQConfig  # noqa: F401
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


device = "cuda"
dtype = torch.bfloat16
SDNQ_BITS = 4  # 4 or 8
repo_id = f"OzzyGT/Flux_3_Image_Klein_sdnq_dynamic_{SDNQ_BITS}bit"
upsampler_id = f"OzzyGT/Flux_3_Image_Klein_pup_sdnq_dynamic_{SDNQ_BITS}bit"
output_dir = "./outputs/flux3_image_klein"
seed = 7
prompt = """A small neighborhood pet shop: a tabby cat on the counter, a golden retriever in a wicker basket and a scarlet macaw perched above an aquarium. The sign over the door reads WHISKERS & WAGS."""

if not seed:
    seed = torch.randint(0, 2**32, (1,)).item()
generator = torch.Generator(device="cpu").manual_seed(seed)

manager = ComponentsManager()
pipe = ModularPipeline.from_pretrained(repo_id, components_manager=manager)
pipe.load_components(dtype=dtype)
pipe.update_components(
    prompt_upsampler=Qwen3_5ForConditionalGeneration.from_pretrained(upsampler_id, dtype=dtype),
    prompt_upsampler_processor=AutoProcessor.from_pretrained(upsampler_id),
)

if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
    pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
    pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)

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

output["images"][0].save(
    os.path.join(output_dir, f"sdnq_{SDNQ_BITS}bit_prompt_upsampler_model_cpu_offload_{seed}.png")
)
