import gc
import os
import subprocess
import sys
import tempfile

import torch
from diffusers import LTX2Pipeline, LTX2VideoTransformer3DModel
from diffusers.hooks import apply_group_offloading
from diffusers.pipelines.ltx2.export_utils import encode_video
from sdnq import SDNQConfig  # noqa: F401
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
from transformers import Gemma3ForConditionalGeneration


onload_device = torch.device("cuda")
offload_device = torch.device("cpu")
torch_dtype = torch.bfloat16

frame_rate = 24
prompt = """Cinematic video with professional lighting, shot with a 35mm lens and shallow depth of field. The scene opens on an extreme close-up of a real car license plate reading “DIFFUSERS,” captured with macro focus, rock music kicks in and builds intensity. The setting is an outdoor wet mountain road, surrounded by dark silhouettes of hills, trees, and winding asphalt. The car is a high-performance modern sports car with large performance tires gripping the rain-soaked road. A deep combustion engine growl builds as the car suddenly accelerates forward."""
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"


def _embed_worker_save(out_path: str) -> None:
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        "Disty0/LTX-2-SDNQ-4bit-dynamic",
        subfolder="text_encoder",
        dtype=torch_dtype,
        device_map="cpu",
    )

    # Text encoding step, can use a separate GPU
    embeds_pipe = LTX2Pipeline.from_pretrained(
        "Lightricks/LTX-2",
        torch_dtype=torch.bfloat16,
        text_encoder=text_encoder,
        transformer=None,
        vae=None,
        audio_vae=None,
        vocoder=None,
    )

    if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
        embeds_pipe.text_encoder = apply_sdnq_options_to_model(embeds_pipe.text_encoder, use_quantized_matmul=True)

    apply_group_offloading(
        embeds_pipe.text_encoder, onload_device=onload_device, offload_type="leaf_level", use_stream=True
    )

    (prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask) = (
        embeds_pipe.encode_prompt(prompt=prompt, negative_prompt=negative_prompt)
    )

    payload = {
        "prompt_embeds": prompt_embeds.detach().cpu(),
        "prompt_attention_mask": prompt_attention_mask.detach().cpu(),
        "negative_prompt_embeds": negative_prompt_embeds.detach().cpu(),
        "negative_prompt_attention_mask": negative_prompt_attention_mask.detach().cpu(),
    }

    # Make sure async stream work is finished before teardown
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    del embeds_pipe
    del text_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.save(payload, out_path)


# --- Subprocess worker mode (same script) ---
if os.environ.get("LTX2_EMBED_WORKER", "0") == "1":
    _embed_worker_save(os.environ["LTX2_EMBED_OUT"])
    raise SystemExit(0)

# --- Parent process: run embed step in a killable subprocess ---
with tempfile.NamedTemporaryFile(prefix="ltx2_embeds_", suffix=".pt", delete=False) as tmp:
    embeds_out_path = tmp.name

env = os.environ.copy()
env["LTX2_EMBED_WORKER"] = "1"
env["LTX2_EMBED_OUT"] = embeds_out_path

embeds_proc = subprocess.Popen([sys.executable, __file__], env=env)

rc = embeds_proc.wait()
if rc != 0:
    raise RuntimeError(f"Embedding worker failed with exit code {rc}")

payload = torch.load(embeds_out_path, map_location="cpu")
try:
    os.remove(embeds_out_path)
except OSError:
    pass

# If you want to forcibly kill it after getting the embeddings (usually already exited):
if embeds_proc.poll() is None:
    embeds_proc.terminate()
    try:
        embeds_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        embeds_proc.kill()

prompt_embeds_clone = payload["prompt_embeds"]
prompt_attention_mask_clone = payload["prompt_attention_mask"]
negative_prompt_embeds_clone = payload["negative_prompt_embeds"]
negative_prompt_attention_mask_clone = payload["negative_prompt_attention_mask"]

del payload
gc.collect()

quit()

transformer = LTX2VideoTransformer3DModel.from_pretrained(
    "Disty0/LTX-2-SDNQ-4bit-dynamic", subfolder="transformer", torch_dtype=torch_dtype, device_map="cpu"
)

denoise_pipe = LTX2Pipeline.from_pretrained(
    "Lightricks/LTX-2",
    transformer=transformer,
    vae=None,
    audio_vae=None,
    text_encoder=None,
    tokenizer=None,
    vocoder=None,
    torch_dtype=torch_dtype,
).to("cuda")

if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
    denoise_pipe.transformer = apply_sdnq_options_to_model(denoise_pipe.transformer, use_quantized_matmul=True)

denoise_pipe.transformer.enable_group_offload(
    onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level", use_stream=True
)

prompt_embeds_clone = prompt_embeds_clone.to("cuda")
prompt_attention_mask_clone = prompt_attention_mask_clone.to("cuda")
negative_prompt_embeds_clone = negative_prompt_embeds_clone.to("cuda")
negative_prompt_attention_mask_clone = negative_prompt_attention_mask_clone.to("cuda")

latents = denoise_pipe(
    prompt_embeds=prompt_embeds_clone,
    prompt_attention_mask=prompt_attention_mask_clone,
    negative_prompt_embeds=negative_prompt_embeds_clone,
    negative_prompt_attention_mask=negative_prompt_attention_mask_clone,
    width=768,
    height=512,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=40,
    guidance_scale=4.0,
    generator=torch.Generator("cuda").manual_seed(42),
    output_type="np",
    return_dict=False,
)
