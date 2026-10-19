# FLUX 3 Image Klein

| Script | Description |
| --- | --- |
| [base_example.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/flux3_image_klein/scripts/base_example.py) | Text-to-image with a bounding box layout in bfloat16, with no offloading |
| [sdnq_example.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/flux3_image_klein/scripts/sdnq_example.py) | SDNQ dynamic quantization (4-bit or 8-bit, toggle via `SDNQ_BITS`), with no offloading |
| [sdnq_group_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/flux3_image_klein/scripts/sdnq_group_offload.py) | SDNQ dynamic quantization (4-bit or 8-bit, toggle via `SDNQ_BITS`) with leaf-level group offloading for the transformer and text encoder, the lowest-VRAM option |
| [prompt_upsampler_model_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/flux3_image_klein/scripts/prompt_upsampler_model_cpu_offload.py) | bfloat16 with the official prompt upsampler, which expands a short request into a full caption, using model CPU offloading |
| [sdnq_prompt_upsampler_model_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/flux3_image_klein/scripts/sdnq_prompt_upsampler_model_cpu_offload.py) | SDNQ pipeline and SDNQ prompt upsampler (4-bit or 8-bit, toggle via `SDNQ_BITS`) in one pipeline with model CPU offloading; tested on 16 GB and 24 GB GPUs |
| [sdnq_prompt_upsampler_two_stage.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/flux3_image_klein/scripts/sdnq_prompt_upsampler_two_stage.py) | SDNQ pipeline and SDNQ prompt upsampler (4-bit or 8-bit, toggle via `SDNQ_BITS`) in two stages: the upsampler writes the caption and is freed, then the image is generated with model CPU offloading; tested on 12 GB (4-bit and 8-bit) and 8 GB (4-bit) GPUs |

> [!NOTE]
> FLUX 3 Image Klein needs full scene captions. All scripts except the `prompt_upsampler` ones skip the pipeline's prompt upsampler step, so pass a caption from the [prompt upsampler](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup), from another LLM using its `system_prompts.json`, or one written following the [prompting guidelines](https://huggingface.co/black-forest-labs/flux-3-image-klein).
