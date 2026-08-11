# Local training runs

This directory stores local training outputs. Run directories are intentionally
ignored by Git because checkpoints, logs, and samples can be large. Keep a
complete copy of any result that matters, including its configuration and
metrics.

## Layout

```text
runs/
└── <run-name>/
    ├── checkpoints/
    │   ├── best.ckpt
    │   └── last.ckpt
    ├── logs/
    ├── resolved-config.yaml
    ├── environment.json
    ├── manifest.yaml
    ├── status.yaml
    └── samples/
        └── <seed-and-count>/
            ├── samples.npz
            ├── generated.parquet
            ├── manifest.yaml
            └── <airport>-trajectories.png
```

Example:

```bash
uv run chabuduo train \
  data/processed/beijing_kinematic_circular-v1 \
  --config src/chabuduo/configs/tcvae_long.yaml \
  --output runs/beijing-kinematic-circular-v1-tcvae-seed-42-1000 \
  --seed 42

uv run chabuduo sample \
  runs/beijing-kinematic-circular-v1-tcvae-seed-42-1000 \
  --count 100 --seed 42
```

Use `best.ckpt` for comparisons; it is selected from validation loss.
`last.ckpt` is useful for debugging or resuming but is not automatically the
best model. Keep the sample seed, count, temperature, and plot with the run.

A remote GPU run is not complete until the whole run directory is copied back
here. Do not report a result from a run that exists only under `/tmp` or only on
the remote machine.
