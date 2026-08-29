# LTX-2.3

LTX-2.3 generates synchronized video and audio. One-stage scripts generate at full resolution in a single pass, while two-stage scripts generate at a reduced resolution first and then refine at full resolution with the distilled model. The conditioning and LoRA scripts rely on the custom pipeline in `pipeline_ltx2_multimodal.py`.

| Script | Description |
| --- | --- |
| [ltx23_one_stage.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2_3/scripts/ltx23_one_stage.py) | Single-stage text-to-video+audio at full resolution with CFG (non-distilled, 30 steps) and group offloading |
| [ltx23_one_stage_sdnq.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2_3/scripts/ltx23_one_stage_sdnq.py) | Single-stage generation with SDNQ 4-bit quantization |
| [ltx23_one_stage_bnb.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2_3/scripts/ltx23_one_stage_bnb.py) | Single-stage generation with BitsAndBytes 4-bit quantization |
| [ltx23_one_stage_distilled.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2_3/scripts/ltx23_one_stage_distilled.py) | Single-stage distilled generation (8 steps, no CFG) with group offloading |
| [ltx23_one_stage_distilled_sdnq.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2_3/scripts/ltx23_one_stage_distilled_sdnq.py) | Single-stage distilled generation with SDNQ 4-bit quantization |
| [ltx23_one_stage_distilled_bnb.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2_3/scripts/ltx23_one_stage_distilled_bnb.py) | Single-stage distilled generation with BitsAndBytes 4-bit quantization |
| [ltx23_two_stages.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2_3/scripts/ltx23_two_stages.py) | Two-stage: low-res non-distilled generation (with CFG) + full-res distilled refinement |
| [ltx23_two_stages_sdnq.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2_3/scripts/ltx23_two_stages_sdnq.py) | Two-stage generation with SDNQ 4-bit quantization |
| [ltx23_two_stages_sdnq_distilled.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2_3/scripts/ltx23_two_stages_sdnq_distilled.py) | Two-stage fully distilled generation with SDNQ 4-bit quantization |
| [t2v_sdnq_4bit_both_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2_3/scripts/t2v_sdnq_4bit_both_cpu_offload.py) | Text-to-video with SDNQ 4-bit quantization and model CPU offloading |
| [t2v_sdnq_4bit_distilled_both_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2_3/scripts/t2v_sdnq_4bit_distilled_both_cpu_offload.py) | Distilled text-to-video with SDNQ 4-bit quantization and model CPU offloading |
| [t2v_sdnq_one_stage_distilled_steps.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2_3/scripts/t2v_sdnq_one_stage_distilled_steps.py) | Single-stage distilled text-to-video with SDNQ 4-bit quantization and group offloading |
| [i2v_sdnq_one_stage_distilled_steps.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2_3/scripts/i2v_sdnq_one_stage_distilled_steps.py) | Single-stage distilled image-to-video with SDNQ 4-bit quantization and group offloading |
| [ltx23_image_audio_conditioning.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2_3/scripts/ltx23_image_audio_conditioning.py) | Image + audio conditioned generation (lip-sync to driving audio) with SDNQ quantization |
| [ltx23_image_audio_conditioning_distilled.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2_3/scripts/ltx23_image_audio_conditioning_distilled.py) | Distilled image + audio conditioned generation with SDNQ quantization |
| [ltx23_ic_lora_sdnq.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2_3/scripts/ltx23_ic_lora_sdnq.py) | IC-LoRA controlled generation with audio conditioning and SDNQ quantization |
| [ltx23_id_lora_sdnq.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2_3/scripts/ltx23_id_lora_sdnq.py) | ID-LoRA generation with reference audio (identity / voice transfer) and SDNQ quantization |
| [ltx23_id_lora_sdnq_distilled_hq.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2_3/scripts/ltx23_id_lora_sdnq_distilled_hq.py) | Two-stage high-quality ID-LoRA generation combining ID and distilled LoRAs with SDNQ quantization |
| [ltx23_video_extension_sdnq.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2_3/scripts/ltx23_video_extension_sdnq.py) | Video extension with the source audio locked for the initial segment and SDNQ quantization |
| [pipeline_ltx2_multimodal.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/ltx2_3/scripts/pipeline_ltx2_multimodal.py) | Custom `LTX2MultiModalPipeline` providing frame/time video+audio conditioning and IC-LoRA control (imported by the conditioning and LoRA scripts above) |
