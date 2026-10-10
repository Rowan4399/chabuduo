"""Export frozen TCVAE posterior parameters, verifying cached latent means."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from chabuduo.models.tcvae import TCVAELightning


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.reference / "manifest.json").read_text())
    checkpoint_hash = digest(args.checkpoint)
    if checkpoint_hash != manifest["tcvae_checkpoint_sha256"]:
        raise ValueError("TCVAE checkpoint differs from the DDPM latent source")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = (
        TCVAELightning.load_from_checkpoint(
            args.checkpoint, map_location=device, weights_only=False
        )
        .to(device)
        .eval()
    )
    module.requires_grad_(False)
    exported = {}
    errors = {}
    hashes = {}
    for offset, split in enumerate(("train", "test")):
        source = args.artifact / f"{split}.npz"
        with np.load(source, allow_pickle=False) as data:
            values = data["values"].astype(np.float32)
            ids = data["trajectory_id"].astype(str)
        with np.load(args.reference / f"{split}.npz") as cached:
            np.testing.assert_array_equal(ids, cached["trajectory_id"].astype(str))
            expected = cached["latent_mu"].copy()
        means, log_variances = [], []
        with torch.inference_mode():
            for start in range(0, len(values), 256):
                batch = torch.from_numpy(values[start : start + 256]).to(device)
                mean, log_variance = module.network.encode(batch.permute(0, 2, 1))
                means.append(mean.cpu().numpy())
                log_variances.append(log_variance.cpu().numpy())
        mean = np.concatenate(means)
        log_variance = np.concatenate(log_variances)
        np.testing.assert_allclose(mean, expected, atol=2e-4, rtol=2e-4)
        noise = np.random.default_rng(args.seed + offset).standard_normal(mean.shape)
        sample = (mean + np.exp(0.5 * log_variance) * noise).astype(np.float32)
        if not np.isfinite(sample).all():
            raise ValueError("Nonfinite posterior samples")
        exported.update(
            {
                f"{split}_mu": mean,
                f"{split}_log_variance": log_variance,
                f"{split}_sample": sample,
                f"{split}_trajectory_id": ids,
            }
        )
        errors[split] = float(np.max(np.abs(mean - expected)))
        hashes[f"{split}.npz"] = digest(source)
        print(
            f"{split}: {mean.shape}, cached mean max error={errors[split]}", flush=True
        )
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "posterior.npz", **exported)
    result = {
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": manifest["tcvae_checkpoint_epoch"],
        "source_artifact": str(args.artifact),
        "source_sha256": hashes,
        "source_script_sha256": digest(Path(__file__)),
        "reference_manifest_sha256": digest(args.reference / "manifest.json"),
        "max_absolute_error_vs_cached_mu": errors,
        "posterior_sample_seeds": {"train": args.seed, "test": args.seed + 1},
        "execution": f"remote frozen encoder inference on {device}; no training",
        "environment": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "lightning")
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf8"
    )
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
