# MiniMax-H3

MiniMax-H3 generates synchronized video and audio from a single text prompt. Unlike the other models here it is a **modular** pipeline (`MiniMaxH3ModularPipeline`), and it runs its transformer over one packed 1-D sequence that holds every modality at once — text, keyframe conditions, target audio and target video rows share a single 40-units-per-second rotary clock, and that shared clock is the audio/video alignment. The conditioner is a Qwen3-VL model, read at the 50th of its 64 decoder layers. The checkpoint is guidance-distilled, so there is no unconditional pass.

The generation canvas is constrained by the checkpoint: a fixed 24 fps, a 768 pixel short edge with a soft `768x1344` area cap, both axes rounded to a multiple of 32, an aspect ratio between 1:4 and 4:1, and a 5 to 15 second duration. `num_frames` is snapped up to the next `17 * n + 5` the video VAE can encode (which maps to `5 * n + 2` latent frames), so a request of 345 frames is 14.4 seconds at 102 latent frames. Audio comes out as stereo at 40 latents per second, decoded by a separate audio autoencoder.

Weights are large enough that offloading is mandatory on a 32 GB card — the bf16 pipeline is well over 50 GB, and even at SDNQ 4-bit the four components total roughly 47 GB, with the Qwen3-VL text encoder rather than the transformer being the single largest one.

| Script | Description |
| --- | --- |
| [t2v_base_example.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/minimax_h3/scripts/t2v_base_example.py) | Basic text-to-video+audio generation with no offloading, from the bfloat16 checkpoint |
| [t2v_sdnq_group_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/minimax_h3/scripts/t2v_sdnq_group_offload.py) | SDNQ (4-bit or 8-bit) with group offloading: block-level streaming for the transformer, leaf-level for the text encoder and video VAE |
| [t2v_sdnq_managed.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/minimax_h3/scripts/t2v_sdnq_managed.py) | SDNQ (4-bit or 8-bit) with diffusers-mm automatic offload management |
| [t2v_sdnq_managed_text_encoder_unload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/minimax_h3/scripts/t2v_sdnq_managed_text_encoder_unload.py) | SDNQ (4-bit or 8-bit) with diffusers-mm, dropping the text encoder once denoising starts to free the most host RAM |
| [t2v_sdnq_managed_two_phase.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/minimax_h3/scripts/t2v_sdnq_managed_two_phase.py) | SDNQ (4-bit or 8-bit) with diffusers-mm, splitting the workflow so the prompt is encoded before the transformer is ever loaded |

The base script is a reference for the plain API rather than something most people can run: bfloat16 MiniMax-H3 is roughly 124 GB of weights, so `pipe.to("cuda")` needs a card that fits all of it. The SDNQ scripts are the ones that run on a 24 to 32 GB card; flip `SDNQ_BITS` between `4` and `8` in each of them.

`t2v_sdnq_managed_two_phase.py` is the lowest-memory option. Because the text encoder and the transformer are never in memory at the same time, peak host RAM is the larger of the two rather than their sum, and the prompt is encoded using barely any VRAM at all.

> [!NOTE]
> MiniMax-H3 support is not in `diffusers` `main` yet, so the pinned `diffusers` git dependency in this repository cannot import `MiniMaxH3ModularPipeline` at the moment.

> [!NOTE]
> The `t2v_sdnq_managed.py`, `t2v_sdnq_managed_text_encoder_unload.py` and `t2v_sdnq_managed_two_phase.py` scripts use the [`diffusers-mm`](https://github.com/asomoza/diffusers-mm) library for automatic offload management. If you're copying a script into your own environment, install it with:
>
> ```bash
> uv add diffusers-mm
> ```
