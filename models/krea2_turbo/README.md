# Krea-2-Turbo

| Script | Description |
| --- | --- |
| [base_t2i_example.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/krea2_turbo/scripts/base_t2i_example.py) | Basic text-to-image generation on the GPU (8 steps, guidance scale 1.0) |
| [t2i_group_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/krea2_turbo/scripts/t2i_group_offload.py) | Text-to-image generation with leaf-level group offloading (streamed) for reduced VRAM usage |
| [t2i_sdnq.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/krea2_turbo/scripts/t2i_sdnq.py) | SDNQ dynamic quantization (4-bit or 8-bit) with quantized matmul for reduced VRAM usage |
| [reference_vision_only.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/krea2_turbo/scripts/reference_vision_only.py) | Reference image conditioning without a LoRA |
| [reference_identity_edit.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/krea2_turbo/scripts/reference_identity_edit.py) | Identity preserving edit with the identity-edit LoRA |
| [reference_style.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/krea2_turbo/scripts/reference_style.py) | Style transfer with the style-reference LoRA |
