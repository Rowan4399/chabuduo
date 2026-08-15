# Training runs

Run directories contain model outputs, not generated configuration files. The
committed experiment YAML remains the only configuration source of truth.

```text
runs/<experiment>-<version>-seed-<seed>/
├── checkpoints/
│   ├── best.ckpt
│   └── last.ckpt
├── logs/
│   └── training/version_*/metrics.csv
├── environment.json
└── samples/
    └── seed-<seed>-n-<count>/
        ├── samples.npz
        ├── generated.parquet
        └── <airport>-trajectories.png
```

Example:

```bash
uv run chabuduo train beijing_tcvae --smoke
uv run chabuduo train beijing_tcvae
uv run chabuduo sample beijing_tcvae
uv run chabuduo plot beijing_tcvae
```

`best.ckpt` is selected using validation loss. Do not substitute `last.ckpt`
without inspecting the validation history. Lightning checkpoints retain the
resolved neural dimensions needed to reload the model; source, preprocessing,
training, and sampling policy remain in `beijing_tcvae.yaml`.
