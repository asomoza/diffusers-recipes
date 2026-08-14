# MiniMax-H3

| Script | Description |
| --- | --- |
| [t2v_base_example.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/minimax_h3/scripts/t2v_base_example.py) | Basic text-to-video+audio generation with no offloading, from the bfloat16 checkpoint |
| [t2v_sdnq_group_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/minimax_h3/scripts/t2v_sdnq_group_offload.py) | SDNQ dynamic quantization (4-bit or 8-bit) with group offloading: block-level streaming for the transformer, leaf-level for the text encoder and video VAE |
| [t2v_sdnq_managed.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/minimax_h3/scripts/t2v_sdnq_managed.py) | SDNQ dynamic quantization (4-bit or 8-bit) with diffusers-mm automatic offload management |
| [t2v_sdnq_managed_text_encoder_unload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/minimax_h3/scripts/t2v_sdnq_managed_text_encoder_unload.py) | SDNQ dynamic quantization (4-bit or 8-bit) with diffusers-mm, dropping the text encoder once denoising starts to free host RAM |
| [t2v_sdnq_managed_two_phase.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/minimax_h3/scripts/t2v_sdnq_managed_two_phase.py) | SDNQ dynamic quantization (4-bit or 8-bit) with diffusers-mm, encoding the prompt before the transformer is ever loaded, the lowest-memory option |

> [!NOTE]
> The `t2v_sdnq_managed.py`, `t2v_sdnq_managed_text_encoder_unload.py` and `t2v_sdnq_managed_two_phase.py` scripts use the [`diffusers-mm`](https://github.com/asomoza/diffusers-mm) library for automatic offload management. They also set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which measurably speeds up the VAE decode. If you're copying a script into your own environment, install it with:
>
> ```bash
> uv add diffusers-mm
> ```
