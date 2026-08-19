import gc
import os

import torch
from diffusers import AutoencoderKLMiniMaxH3, ModularPipeline
from diffusers.utils.export_utils import encode_video
from diffusers_mm import ModelManager, managed
from sdnq import SDNQConfig  # noqa: F401
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model


# Use only on CUDA
if torch.version.hip is None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# patch until fixed upstream, the fp32 pin covers the whole VAE and ignores dtype
AutoencoderKLMiniMaxH3._keep_in_fp32_modules = []

device = "cuda"
dtype = torch.bfloat16
SDNQ_BITS = 4  # 4 or 8
repo_id = f"OzzyGT/MiniMax_H3_sdnq_{SDNQ_BITS}bit_pruned"
output_dir = "./outputs/minimax_h3"
prompt = """integrated_multimodal_description: Live-action overhead shot looking straight down at a polished dark wooden table under bright even studio light. Many white dominoes stand on end in tightly packed lines, and those lines are laid out so that together they form large block capital letters spelling "DIFFUSERS". At the start the camera is close in over the very first domino at the left end of the word, so only that domino and a few of its neighbours fill the frame. The dominoes spelling "DIFFUSERS" are the only text in the shot. [0s-0.8s] Everything is completely still and a single index finger rests lightly against the first standing domino. [0.8s-1.2s] The finger tips the first domino over and then lifts away out of the frame. [1.2s-4.0s] The dominoes topple one after another in one continuous chain that runs away along the lines, and at the same time the camera pulls straight back overhead at a steady slow speed, so that more and more of the layout comes into view as the falling chain travels through it. [4.0s-5.1s] All action is over. Every domino is lying flat, the camera has stopped moving, and the fallen white dominoes clearly spell "DIFFUSERS" across the dark table. Nothing moves and nothing else happens for the rest of the shot.

overall_soundscape: A quiet room tone for the first beat. A single soft tap as the finger tips the first domino over, then immediately a rapid cascade of sharp wooden clicks as each domino strikes the next, running away from the microphone in a steady quick rhythm. The cascade finishes as the last domino lands, leaving only the quiet room tone for the rest of the shot.

non_diegetic_music: No music during the first beat. A light playful pizzicato string figure enters as the chain starts falling and keeps pace with the cascade. Over the last second it resolves onto one clean final chord that lands on the last frame."""

workflow = ModularPipeline.from_pretrained(repo_id, trust_remote_code=True).blocks.get_workflow("t2va")

conditioner = workflow.sub_blocks.pop("text_encoder").init_pipeline(repo_id)
conditioner.load_components(dtype=dtype, trust_remote_code=True)

if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
    conditioner.text_encoder = apply_sdnq_options_to_model(conditioner.text_encoder, use_quantized_matmul=True)

conditioner = managed(conditioner, strategy="auto", device=device)
# encodes the prompt with only the text encoder in memory, then frees it before anything else loads
state = conditioner(prompt=prompt)

conditioner.mm.clear()
del conditioner
gc.collect()
torch.cuda.empty_cache()
ModelManager.release_host_cache()

rest = workflow.init_pipeline(repo_id)
rest.load_components(dtype=dtype, trust_remote_code=True)

if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
    rest.transformer = apply_sdnq_options_to_model(rest.transformer, use_quantized_matmul=True)

rest = managed(rest, strategy="auto", device=device)

results = rest(
    state=state,
    height=480,
    width=864,
    num_frames=124,
    num_inference_steps=20,
    generator=torch.Generator("cpu").manual_seed(2236370900),
    output=["videos", "audio", "sampling_rate"],
)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

encode_video(
    results.get("videos")[0],
    fps=24,
    audio=results.get("audio")[0],
    audio_sample_rate=results.get("sampling_rate"),
    output_path=os.path.join(output_dir, f"t2v_sdnq_{SDNQ_BITS}bit_managed_two_phase.mp4"),
)
