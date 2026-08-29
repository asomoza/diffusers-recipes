# FLUX.2-dev

| Script | Description |
| --- | --- |
| [base_t2i_example.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/flux2_dev/scripts/base_t2i_example.py) | Basic text-to-image generation with model CPU offloading |
| [t2i_managed.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/flux2_dev/scripts/t2i_managed.py) | Text-to-image generation with diffusers-mm automatic offload management |
| [t2i_sdnq_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/flux2_dev/scripts/t2i_sdnq_cpu_offload.py) | SDNQ dynamic quantization (4-bit or 8-bit) with model CPU offloading for reduced VRAM usage |
| [t2i_sdnq_managed.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/flux2_dev/scripts/t2i_sdnq_managed.py) | SDNQ dynamic quantization (4-bit or 8-bit) with diffusers-mm automatic offload management |

> [!NOTE]
> The `t2i_managed.py` and `t2i_sdnq_managed.py` scripts use the [`diffusers-mm`](https://github.com/asomoza/diffusers-mm) library for automatic offload management. If you're copying a script into your own environment, install it with:
>
> ```bash
> uv add diffusers-mm
> ```
