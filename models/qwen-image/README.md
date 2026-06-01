# Qwen-Image

| Script | Description |
| --- | --- |
| [base_example.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen-image/scripts/base_example.py) | Basic text-to-image generation |
| [model_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen-image/scripts/model_cpu_offload.py) | Text-to-image generation with model CPU offloading |
| [bnb_4bit_both_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen-image/scripts/bnb_4bit_both_cpu_offload.py) | Text-to-image with BitsAndBytes 4-bit quantized transformer and text encoder, plus model CPU offloading |
| [pipeline_leaf_group_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen-image/scripts/pipeline_leaf_group_offload.py) | Text-to-image with leaf-level group offloading for low VRAM usage |
| [pipeline_leaf_stream_group_offload_record.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen-image/scripts/pipeline_leaf_stream_group_offload_record.py) | Text-to-image with leaf-level group offloading using CUDA stream prefetching |
