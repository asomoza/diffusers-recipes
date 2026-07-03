# Qwen-Image-2512

| Script | Description |
| --- | --- |
| [base_example.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen_image_2512/scripts/base_example.py) | Basic text-to-image generation |
| [model_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen_image_2512/scripts/model_cpu_offload.py) | Text-to-image generation with model CPU offloading |
| [bnb_4bit_both_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen_image_2512/scripts/bnb_4bit_both_cpu_offload.py) | Text-to-image with BitsAndBytes 4-bit quantized transformer and text encoder, plus model CPU offloading |
| [pipeline_leaf_group_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen_image_2512/scripts/pipeline_leaf_group_offload.py) | Text-to-image with leaf-level group offloading for low VRAM usage |
| [pipeline_leaf_stream_group_offload_record.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen_image_2512/scripts/pipeline_leaf_stream_group_offload_record.py) | Text-to-image with leaf-level group offloading using CUDA stream prefetching |
| [sdnq.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen_image_2512/scripts/sdnq.py) | Text-to-image with SDNQ dynamic quantization (4-bit or 8-bit), toggle via `SDNQ_BITS` |
| [sdnq_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen_image_2512/scripts/sdnq_cpu_offload.py) | SDNQ dynamic quantization (4-bit or 8-bit) plus model CPU offloading for lower VRAM usage |
