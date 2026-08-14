import os

import torch
from diffusers import AutoencoderKLMiniMaxH3, ModularPipeline
from diffusers.utils.export_utils import encode_video


# patch until fixed upstream, the fp32 pin covers the whole VAE and ignores dtype
AutoencoderKLMiniMaxH3._keep_in_fp32_modules = []

device = "cuda"
dtype = torch.bfloat16
repo_id = "MiniMaxAI/MiniMax-H3"
output_dir = "./outputs/minimax_h3"
prompt = """integrated_multimodal_description: [Shot 1] Live-action, cinematic food commercial, a macro shot centered on a matte black slate plate against a warm charcoal backdrop, lit by a hard key light from the upper left. In slow motion a toasted brioche bun heel drops into frame and settles on the slate, followed by a thick flame-grilled beef patty that lands with a shudder and throws a fine spray of juice through the light. A square of orange cheddar falls next and immediately begins to slump over the patty edges. A crisp green lettuce leaf and two thick tomato slices sail in and settle into place. A male announcer with a deep, warm, gravelly voice (S1) says in an off-screen voiceover: <d>[English] Stacked to order. Never frozen.</d> The camera pushes in with small amplitude at slow speed as the top bun crown descends and lands, compressing the stack slightly. [Shot 2] At 00:04.500, the camera cuts to a wider shot and tilts up with small amplitude at slow speed from the finished burger to a hand-painted wooden sign reading "COPPER CREEK BURGER CO." mounted above a black chalkboard menu reading "DOUBLE SMASH 9.50", "BACON STACK 11.00", and "MALTED SHAKE 5.00", while the announcer (S1) says in an off-screen voiceover: <d>[English] Copper Creek. Since nineteen fifty-two.</d>

overall_soundscape: A low warm room tone runs underneath while each component lands with its own weight: the soft give of the bun, a heavy wet slap from the patty, and the faint crackle of rendered fat. A brief sizzle rises as the cheese slumps, followed by the light crunch of lettuce and the dull knock of the top bun settling.

non_diegetic_music: An upright bass and brushed snare pattern at a medium-slow tempo, joined by a single clean electric guitar figure that lands on the beat of each falling ingredient and resolves into a held chord at the end."""

pipe = ModularPipeline.from_pretrained(repo_id, workflow="t2va")
pipe.load_components(dtype=dtype)
pipe.to(device)

state = pipe(
    prompt=prompt,
    height=768,
    width=1344,
    num_frames=124,
    num_inference_steps=30,
    generator=torch.Generator("cpu").manual_seed(42),
    output=["videos", "audio", "sampling_rate"],
)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

encode_video(
    state.get("videos")[0],
    fps=24,
    audio=state.get("audio")[0],
    audio_sample_rate=state.get("sampling_rate"),
    output_path=os.path.join(output_dir, "t2v_base_example.mp4"),
)
