# Qwen-Image-2.1

| Script | Description |
| --- | --- |
| [base_example.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen_image_2_1/scripts/base_example.py) | Text-to-image in bfloat16 with model CPU offloading |
| [sdnq_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen_image_2_1/scripts/sdnq_cpu_offload.py) | SDNQ dynamic quantization (4-bit or 8-bit, toggle via `SDNQ_BITS`) plus model CPU offloading |
| [base_example_vae_fix.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen_image_2_1/scripts/base_example_vae_fix.py) | Same as `base_example.py`, with the seam-free strip VAE decode (`qwen_image_21_vae_strips.py`) instead of tiling |
| [turbo_base_example.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen_image_2_1/scripts/turbo_base_example.py) | Qwen-Image-2.1-Turbo text-to-image in bfloat16 with its saved 8-step schedule and model CPU offloading |
| [turbo_sdnq_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen_image_2_1/scripts/turbo_sdnq_cpu_offload.py) | Qwen-Image-2.1-Turbo with SDNQ dynamic quantization (4-bit or 8-bit, toggle via `SDNQ_BITS`) plus model CPU offloading |
| [turbo_base_example_vae_fix.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen_image_2_1/scripts/turbo_base_example_vae_fix.py) | Same as `turbo_base_example.py`, with the seam-free strip VAE decode (`qwen_image_21_vae_strips.py`) instead of tiling |
