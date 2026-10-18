# MOSS-Image

| Script | Description |
| --- | --- |
| [base_example.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/moss_image/scripts/base_example.py) | Text-to-image with a bounding box layout in bfloat16, with no offloading (needs more than 32 GB of VRAM at 2048x1152) |
| [group_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/moss_image/scripts/group_offload.py) | bfloat16 with leaf-level group offloading for the transformer and text encoder |
| [sdnq_example.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/moss_image/scripts/sdnq_example.py) | SDNQ dynamic quantization (4-bit or 8-bit, toggle via `SDNQ_BITS`), with no offloading |
| [sdnq_group_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/moss_image/scripts/sdnq_group_offload.py) | SDNQ dynamic quantization (4-bit or 8-bit, toggle via `SDNQ_BITS`) with leaf-level group offloading for the transformer and text encoder, the lowest-VRAM option |

> [!NOTE]
> MOSS-Image runs as a Modular Diffusers pipeline built from the [MOSS-Image custom blocks](https://huggingface.co/OzzyGT/moss_image_custom_blocks). `trust_remote_code=True` runs the code of the blocks repository and `trust_components_code=True` runs the code shipped in the model repository, so only use them with repositories you trust. The scripts use 28 steps, the blocks' default; the official recommendation is 50 steps or more. For the layout prompt format and prompting advice, see the [user guide](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md).
