# FLUX.2-klein-9B

| Script | Description |
| --- | --- |
| [base_example.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/flux2_klein_9B/scripts/base_example.py) | Basic text-to-image generation with model CPU offloading (4 steps) |
| [t2i_bnb_4bit_both_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/flux2_klein_9B/scripts/t2i_bnb_4bit_both_cpu_offload.py) | Text-to-image with BitsAndBytes 4-bit quantized transformer and text encoder, plus model CPU offloading |
| [t2i_sdnq_4bit_both_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/flux2_klein_9B/scripts/t2i_sdnq_4bit_both_cpu_offload.py) | Text-to-image with SDNQ 4-bit dynamic quantization (transformer and text encoder) and model CPU offloading |
| [i2i_sdnq_4bit_both_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/flux2_klein_9B/scripts/i2i_sdnq_4bit_both_cpu_offload.py) | Image-to-image with SDNQ 4-bit dynamic quantization and model CPU offloading |
| [i2i_multiple_sdnq_4bit_both_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/flux2_klein_9B/scripts/i2i_multiple_sdnq_4bit_both_cpu_offload.py) | Image-to-image with multiple input images, SDNQ 4-bit dynamic quantization and model CPU offloading |
