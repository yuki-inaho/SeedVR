# Chunked / continuous inference runner for SeedVR2-3B.
#
# The stock projects/inference_seedvr2_3b.py reads a whole clip and runs it in a
# single forward — so VRAM scales with total frame count and long sequences OOM.
# This runner loads the model ONCE and processes the input in 4n+1 frame windows
# (the temporal unit SeedVR2 needs), then concatenates the restored frames into a
# single output video. Window size / overlap / resolution are configurable so a
# long sequence can be restored continuously on a single 32GB GPU.
#
# Launch with torchrun (model init uses distributed):
#   PYTHONPATH=$PWD uv run torchrun --nproc-per-node=1 --master_port=29583 \
#     projects/inference_seedvr2_3b_chunked.py \
#     --input raw/clip.mp4 --output restored/clip.mp4 \
#     --res_h 1024 --res_w 768 --chunk 17 --overlap 0

import argparse
import gc
import os

import mediapy
import torch
from einops import rearrange
from tqdm import tqdm
from torchvision.io import read_image
from torchvision.io.video import read_video
from torchvision.transforms import Compose, Lambda, Normalize

from data.image.transforms.divisible_crop import DivisibleCrop
from data.image.transforms.na_resize import NaResize
from data.video.transforms.rearrange import Rearrange
from common.distributed import get_device
from common.seed import set_seed

# Reuse the verified single-clip helpers.
from projects.inference_seedvr2_3b import configure_runner, generation_step, is_image_file

if os.path.exists("./projects/video_diffusion_sr/color_fix.py"):
    from projects.video_diffusion_sr.color_fix import wavelet_reconstruction
    use_colorfix = True
else:
    use_colorfix = False
    print("Note: color_fix not available; output without color fix.")


def cut_videos(videos, sp_size):
    """Pad temporal dim so (t-1) % (4*sp_size) == 0 (same as stock script)."""
    t = videos.size(1)
    if t == 1:
        return videos
    if t <= 4 * sp_size:
        padding = [videos[:, -1].unsqueeze(1)] * (4 * sp_size - t + 1)
        return torch.cat([videos, *padding], dim=1)
    if (t - 1) % (4 * sp_size) == 0:
        return videos
    padding = [videos[:, -1].unsqueeze(1)] * (4 * sp_size - ((t - 1) % (4 * sp_size)))
    videos = torch.cat([videos, *padding], dim=1)
    assert (videos.size(1) - 1) % (4 * sp_size) == 0
    return videos


def read_input_frames(path):
    """Return ([T, C, H, W] float in 0..1, fps_or_None)."""
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if is_image_file(f))
        frames = [read_image(os.path.join(path, f)) for f in files]
        return torch.stack(frames, 0).float() / 255.0, None
    vid, _, info = read_video(path, output_format="TCHW")
    return vid.float() / 255.0, info.get("video_fps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="video file OR a folder of frames")
    ap.add_argument("--output", required=True, help="output video path (e.g. restored/clip.mp4)")
    ap.add_argument("--res_h", type=int, default=1024)
    ap.add_argument("--res_w", type=int, default=768)
    ap.add_argument("--chunk", type=int, default=17, help="frames per window, must be 4n+1")
    ap.add_argument("--overlap", type=int, default=0, help="frames shared between windows (dropped on stitch)")
    ap.add_argument("--sp_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--out_fps", type=float, default=None)
    args = ap.parse_args()

    assert (args.chunk - 1) % 4 == 0, "--chunk must be 4n+1 (5, 9, 13, 17, 25, ...)"
    assert 0 <= args.overlap < args.chunk, "--overlap must be in [0, chunk)"

    runner = configure_runner(args.sp_size)
    # SeedVR2 = single-step, cfg off (mirror stock generation_loop defaults).
    runner.config.diffusion.cfg.scale = 1.0
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion()
    set_seed(args.seed, same_across_ranks=True)

    video, fps = read_input_frames(args.input)
    fps = args.out_fps or fps or 8.0
    total = video.size(0)
    print(f"Input frames: {total}, fps={fps}, window={args.chunk}, overlap={args.overlap}")

    transform = Compose([
        NaResize(resolution=(args.res_h * args.res_w) ** 0.5, mode="area", downsample_only=False),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Normalize(0.5, 0.5),
        Rearrange("t c h w -> c t h w"),
    ])

    text_embeds = {
        "texts_pos": [torch.load("pos_emb.pt")],
        "texts_neg": [torch.load("neg_emb.pt")],
    }
    for key in ("texts_pos", "texts_neg"):
        text_embeds[key] = [e.to(get_device()) for e in text_embeds[key]]

    step = args.chunk - args.overlap
    out_frames = []
    starts = list(range(0, total, step))
    n_chunks = len(starts)
    for ci, s in enumerate(tqdm(starts, desc="chunks", unit="chunk")):
        e = min(s + args.chunk, total)
        clip = video[s:e]
        t_in = clip.size(0)
        tqdm.write(f"[chunk {ci + 1}/{n_chunks}] frames {s}..{e - 1} ({t_in})")

        cond = transform(clip.to(get_device()))          # [C, t, H, W]
        input_video = rearrange(cond, "c t h w -> t c h w")
        cond_cut = cut_videos(cond, args.sp_size)

        runner.dit.to("cpu")
        runner.vae.to(get_device())
        latents = runner.vae_encode([cond_cut])
        runner.vae.to("cpu")
        runner.dit.to(get_device())

        sample = generation_step(runner, text_embeds, cond_latents=latents)[0]  # [t, c, h, w]
        sample = sample[:t_in]
        if use_colorfix:
            sample = wavelet_reconstruction(sample.to("cpu"), input_video[:sample.size(0)].to("cpu"))
        else:
            sample = sample.to("cpu")

        # Drop overlap frames on every chunk after the first to avoid duplicates.
        if ci > 0 and args.overlap > 0:
            sample = sample[args.overlap:]
        out_frames.append(sample)

        runner.dit.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        runner.dit.to(get_device())

    frames = torch.cat(out_frames, dim=0)                # [N, c, h, w]
    frames = rearrange(frames, "t c h w -> t h w c")
    frames = frames.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).round().to(torch.uint8).numpy()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    mediapy.write_video(args.output, frames, fps=fps)
    print(f"Wrote {frames.shape[0]} frames -> {args.output}")


if __name__ == "__main__":
    main()
