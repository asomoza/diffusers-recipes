# LTX-2

LTX-2 generates synchronized video and audio. The two-stage scripts generate at a lower resolution first and then refine at full resolution with the distilled model.

| Script | Description |
| --- | --- |
| [t2v_base_example.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2/scripts/t2v_base_example.py) | Basic text-to-video (with audio) generation with model CPU offloading |
| [t2v_bnb_4bit_both_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2/scripts/t2v_bnb_4bit_both_cpu_offload.py) | Text-to-video with BitsAndBytes 4-bit quantized transformer and text encoder, plus model CPU offloading |
| [t2v_sdnq_4bit_both_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2/scripts/t2v_sdnq_4bit_both_cpu_offload.py) | Text-to-video with SDNQ 4-bit dynamic quantization and model CPU offloading |
| [t2v_group_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2/scripts/t2v_group_offload.py) | Text-to-video with group offloading for low VRAM usage |
| [t2v_layerwise.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2/scripts/t2v_layerwise.py) | Text-to-video with FP8 layerwise casting and model CPU offloading |
| [t2v_2_stages_bnb_4bit_quality_steps.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2/scripts/t2v_2_stages_bnb_4bit_quality_steps.py) | Two-stage text-to-video (base generation + distilled refinement) with BitsAndBytes 4-bit quantization and group offloading |
| [t2v_2_stages_distilled_bnb_4bit_quality_steps.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2/scripts/t2v_2_stages_distilled_bnb_4bit_quality_steps.py) | Two-stage fully distilled text-to-video with BitsAndBytes 4-bit quantization and group offloading |
| [t2v_2_stages_distilled_sdnq_4bit_quality_steps.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2/scripts/t2v_2_stages_distilled_sdnq_4bit_quality_steps.py) | Two-stage fully distilled text-to-video with SDNQ 4-bit quantization and group offloading |
| [i2v_bnb_4bit_both_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2/scripts/i2v_bnb_4bit_both_cpu_offload.py) | Image-to-video with BitsAndBytes 4-bit quantization and model CPU offloading |
| [i2v_sdnq_4bit_both_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2/scripts/i2v_sdnq_4bit_both_cpu_offload.py) | Image-to-video with SDNQ 4-bit dynamic quantization and model CPU offloading |
| [i2v_gguf.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2/scripts/i2v_gguf.py) | Image-to-video with a GGUF-quantized transformer (Q4_K_M) and model CPU offloading |
| [i2v_group_offload_leaf_stream.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2/scripts/i2v_group_offload_leaf_stream.py) | Image-to-video with leaf-level group offloading and CUDA stream prefetching |
| [i2v_sdnq_4bit_both_group_offload_leaf_stream.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2/scripts/i2v_sdnq_4bit_both_group_offload_leaf_stream.py) | Image-to-video with SDNQ 4-bit quantization and leaf-level group offloading with stream prefetching |
| [i2v_2_stages_distilled_lora_bnb_4bit.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2/scripts/i2v_2_stages_distilled_lora_bnb_4bit.py) | Two-stage image-to-video using the distilled LoRA, BitsAndBytes 4-bit quantization and model CPU offloading |
| [i2v_2_stages_distilled_transformer_bnb_4bit.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2/scripts/i2v_2_stages_distilled_transformer_bnb_4bit.py) | Two-stage image-to-video with a distilled BitsAndBytes 4-bit transformer and model CPU offloading |
| [i2v_2_stages_distilled_transformer_bnb_4bit_10s.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2/scripts/i2v_2_stages_distilled_transformer_bnb_4bit_10s.py) | Same as above, generating a 10-second clip |
