"""Compare the saved RWY14 and global Zurich DDPM epoch-500 settings."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

rwy14 = Path("runs/zurich_rwy14_ddpm_unconditional-v1-seed-42")
global_run = Path("runs/zurich_ddpm_temporal_unconditional-v1-seed-42")
checkpoint_paths = {
    "rwy14": rwy14 / "resume_sources/epoch-500/last.pt",
    "global": global_run / "resume_sources/epoch-500/checkpoints/last.pt",
}
checkpoints = {
    name: torch.load(path, map_location="cpu", weights_only=False)
    for name, path in checkpoint_paths.items()
}
architecture_keys = [
    "architecture", "feature_dim", "hidden_dim", "diffusion_steps", "time_dim",
    "kernel_size", "dilations", "dropout", "use_phase",
]
details = {}
for name, checkpoint in checkpoints.items():
    details[name] = {
        "path": str(checkpoint_paths[name]),
        "sha256": hashlib.sha256(checkpoint_paths[name].read_bytes()).hexdigest(),
        "epoch": checkpoint["epoch"],
        "architecture": {key: checkpoint[key] for key in architecture_keys},
        "optimizer": [
            {key: value for key, value in group.items() if key != "params"}
            for group in checkpoint["optimizer_state_dict"]["param_groups"]
        ],
        "scheduler": checkpoint["scheduler_state_dict"],
        "ema_present": checkpoint.get("ema_model_state_dict") is not None,
        "scaler": checkpoint.get("scaler_state_dict"),
    }
config_paths = {
    "rwy14": rwy14 / (
        "resume_sources/epoch-500/zurich_rwy14_ddpm_unconditional_epoch500.yaml"
    ),
    "global": global_run / "configs/zurich_ddpm_temporal_unconditional.yaml",
}
configs = {
    name: yaml.safe_load(path.read_text()) for name, path in config_paths.items()
}
states = {name: checkpoint["model_state_dict"]
          for name, checkpoint in checkpoints.items()}
shapes = {
    name: {key: tuple(value.shape) for key, value in state.items()}
    for name, state in states.items()
}
report = {
    "checkpoints": details,
    "architecture_equal": (
        details["rwy14"]["architecture"] == details["global"]["architecture"]
    ),
    "state_keys_and_shapes_equal": shapes["rwy14"] == shapes["global"],
    "state_tensor_count": len(shapes["rwy14"]),
    "optimizer_hyperparameters_equal": (
        details["rwy14"]["optimizer"] == details["global"]["optimizer"]
    ),
    "diffusion_buffers_equal": {
        key: torch.equal(states["rwy14"][key], states["global"][key])
        for key in states["rwy14"]
        if "beta" in key.lower() or "alpha" in key.lower()
    },
    "yaml_sections_equal": {
        section: configs["rwy14"][section] == configs["global"][section]
        for section in ["model", "optimizer", "training", "sampling"]
    },
    "first_500_epochs": {},
    "environment": {},
}
for name, run in [("rwy14", rwy14), ("global", global_run)]:
    metrics = pd.read_csv(run / "logs/metrics.csv")
    metrics = metrics.loc[metrics.epoch <= 500]
    report["first_500_epochs"][name] = {
        "epochs": len(metrics),
        "learning_rates": metrics.learning_rate.value_counts().to_dict(),
        "validation_minimum": metrics.loc[
            metrics.validation_loss.idxmin(), ["epoch", "validation_loss"]
        ].to_dict(),
    }
    report["environment"][name] = json.loads(
        (run / "environment.json").read_text()
    )
output = Path(
    "runs/zurich-rwy14-ddpm-resume700-20261008/architecture_comparison.json"
)
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
print(json.dumps(report, indent=2))
assert report["architecture_equal"]
assert report["state_keys_and_shapes_equal"]
assert report["optimizer_hyperparameters_equal"]
assert all(report["yaml_sections_equal"].values())
assert all(report["diffusion_buffers_equal"].values())
print(f"OUTPUT={output.resolve()}")
