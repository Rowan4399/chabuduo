"""Validate the encoded video and summarize the saved, unmodified outputs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path.cwd() / "runs/latent_denoising_video-v1-seed-42"
RECORDING = ROOT / "recording-n-100-steps-1000-seed-42"


def main() -> None:
    with np.load(RECORDING / "denoising_states.npz") as data:
        history = data["standardized_latent"]
        ids = data["sample_id"]
        xy = data["projected_xy"]
        bounds = data["video_axis_bounds"]
        assert history.shape == (1001, 100, 64)
        assert xy.shape == (1001, 100, 2)
        assert np.isfinite(history).all()
        assert (xy >= bounds[0]).all() and (xy <= bounds[1]).all()
    with np.load(RECORDING / "generated_100_trajectories.npz") as data:
        assert np.array_equal(history[-1], data["generated_latent_standardized"])
        assert np.array_equal(ids, data["sample_id"])
        values = data["values"]
        assert values.shape == (100, 200, 4)
        assert np.isfinite(values).all()

    reader = imageio_ffmpeg.read_frames(str(RECORDING / "latent_denoising_100.mp4"))
    metadata = next(reader)
    reader.close()
    assert metadata["size"] == (1920, 1080)
    assert metadata["fps"] == 50
    count, duration = imageio_ffmpeg.count_frames_and_secs(
        str(RECORDING / "latent_denoising_100.mp4")
    )
    assert count == 1200, count
    assert abs(duration - 24) < 0.05, duration
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-xerror",
            "-i",
            str(RECORDING / "latent_denoising_100.mp4"),
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
    )

    sheet = Image.new("RGB", (1920, 720), "white")
    for index, step in enumerate(range(0, 1001, 200)):
        with Image.open(RECORDING / "frames" / f"step_{step:04d}.png") as frame:
            sheet.paste(
                frame.resize((640, 360), Image.Resampling.LANCZOS),
                ((index % 3) * 640, (index // 3) * 360),
            )
    sheet.save(RECORDING / "contact_sheet.jpg", quality=95)

    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False})
    colors = ["#DE6A26", "#B73D68", "#7453B9", "#087F8C", "#3872B4", "#807029"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), facecolor="#FAFAF7")
    labels = [
        "展开后的航迹角 / degree",
        "地速 / knot",
        "MSL 高度 / ft",
        "经过时间 / s",
    ]
    for feature, axis in enumerate(axes.flat):
        axis.set_facecolor("#FAFAF7")
        for sample in range(6, 100):
            axis.plot(values[sample, :, feature], color="#607180", alpha=0.24, lw=0.8)
        for sample, color in enumerate(colors):
            axis.plot(
                values[sample, :, feature],
                color=color,
                lw=1.6,
                label=f"{sample + 1:03d}",
            )
        axis.set_ylabel(labels[feature])
        axis.set_xlabel("航迹采样点序号（按时间顺序）")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(ncol=6, fontsize=9)
    fig.suptitle("同一批 100 条生成航迹：最终潜变量解码后的原始特征", fontsize=20)
    fig.text(
        0.5,
        0.02,
        "编号和高亮颜色与视频一致；未裁剪、未修正时间、未重建地理位置。",
        ha="center",
        fontsize=11,
        color="#61717C",
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.94))
    fig.savefig(RECORDING / "decoded_trajectory_features.png", dpi=120)
    plt.close(fig)
    result = {
        "video_full_decode_passed": True,
        "video_frames": count,
        "video_duration_seconds": duration,
        "all_100_particles_inside_fixed_view": True,
        "sample_ids_match_decoded_trajectories": True,
        "terminal_latents_exactly_match_saved_samples": True,
        "raw_decoded_decreasing_time_increments": int(
            (np.diff(values[:, :, 3]) < 0).sum()
        ),
        "raw_decoded_speed_range_knots": [
            float(values[:, :, 1].min()),
            float(values[:, :, 1].max()),
        ],
    }
    (RECORDING / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(RECORDING)


if __name__ == "__main__":
    main()
