"""Read-only diagnostics for the learned K=4 VampPrior components."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from chabuduo.artifacts import SplitData
from chabuduo.models.tcvae import TCVAELightning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--prior-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def gaussian_log_prob(
    value: torch.Tensor,
    mean: torch.Tensor,
    log_variance: torch.Tensor,
) -> torch.Tensor:
    return -0.5 * (
        math.log(2.0 * math.pi)
        + log_variance
        + (value - mean).square() / log_variance.exp()
    ).sum(dim=-1)


def distribution_summary(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:
    center = reference.mean(dim=0)
    scale = reference.std(dim=0).clamp_min(1e-6)
    standardized = (candidate - center) / scale
    return {
        "latent_norm_mean": float(candidate.norm(dim=1).mean()),
        "latent_norm_std": float(candidate.norm(dim=1).std()),
        "diagonal_mahalanobis_mean": float(standardized.square().sum(dim=1).mean()),
        "fraction_with_any_dimension_outside_3sigma": float(
            (standardized.abs() > 3.0).any(dim=1).float().mean()
        ),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = TCVAELightning.load_from_checkpoint(
        args.checkpoint, map_location=device
    ).to(device).eval()
    train = SplitData.load(args.train)

    posterior_means: list[torch.Tensor] = []
    posterior_samples: list[torch.Tensor] = []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.inference_mode():
        for start in range(0, len(train.values), args.batch_size):
            value = torch.from_numpy(
                train.values[start : start + args.batch_size]
            ).to(device)
            mean, log_variance = module.network.encode(value.permute(0, 2, 1))
            noise = torch.randn(
                mean.shape, generator=generator, device=device, dtype=mean.dtype
            )
            posterior_means.append(mean)
            posterior_samples.append(mean + torch.exp(0.5 * log_variance) * noise)

        posterior_mean = torch.cat(posterior_means)
        posterior_sample = torch.cat(posterior_samples)
        component_mean, component_log_variance = module.network.prior_parameters()
        log_component = gaussian_log_prob(
            posterior_sample[:, None, :],
            component_mean[None, :, :],
            component_log_variance[None, :, :],
        )
        responsibilities = torch.softmax(log_component, dim=1)
        assignments = responsibilities.argmax(dim=1)
        counts = torch.bincount(assignments, minlength=len(component_mean))

        selected = torch.randint(
            len(component_mean),
            (args.prior_samples,),
            generator=generator,
            device=device,
        )
        noise = torch.randn(
            (args.prior_samples, component_mean.shape[1]),
            generator=generator,
            device=device,
        )
        prior_sample = component_mean[selected] + torch.exp(
            0.5 * component_log_variance[selected]
        ) * noise

        pairwise_distance = torch.cdist(component_mean, component_mean)
        off_diagonal = pairwise_distance[
            ~torch.eye(len(component_mean), dtype=torch.bool, device=device)
        ]

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "train_samples": int(len(train.values)),
        "latent_dimension": int(component_mean.shape[1]),
        "components": int(len(component_mean)),
        "component_hard_assignment_counts": counts.cpu().tolist(),
        "component_hard_assignment_fractions": (
            counts.float() / counts.sum()
        ).cpu().tolist(),
        "responsibility_mean_max": float(responsibilities.max(dim=1).values.mean()),
        "responsibility_low_confidence_fraction": float(
            (responsibilities.max(dim=1).values < 0.6).float().mean()
        ),
        "component_mean_norms": component_mean.norm(dim=1).cpu().tolist(),
        "component_average_standard_deviation": torch.exp(
            0.5 * component_log_variance
        ).mean(dim=1).cpu().tolist(),
        "component_rms_standard_deviation": torch.exp(
            component_log_variance
        ).mean(dim=1).sqrt().cpu().tolist(),
        "component_mean_pairwise_distance_min": float(off_diagonal.min()),
        "component_mean_pairwise_distance_mean": float(off_diagonal.mean()),
        "posterior_mean": distribution_summary(posterior_sample, posterior_mean),
        "posterior_sample": distribution_summary(posterior_sample, posterior_sample),
        "vampprior_sample": distribution_summary(posterior_sample, prior_sample),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
