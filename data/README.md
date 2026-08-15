# Data artifacts

The committed experiment YAML is the only configuration file. Data directories
contain files produced by one explicit CLI stage.

```text
data/
├── raw/        # staged source Parquet
├── interim/    # selected variable-length Traffic
└── processed/  # ready-to-train tensors
```

## Beijing

```bash
uv run chabuduo get beijing_tcvae --source /path/to/feiyou-root
uv run chabuduo preprocess beijing_tcvae --max-workers 4
uv run chabuduo prepare beijing_tcvae
uv run chabuduo validate beijing_tcvae
```

`beijing_tcvae` and `beijing_ddpm` reuse the same raw source and selected
landing cache when their source and preprocessing settings match.

## Zurich

```bash
uv run chabuduo get zurich_tcvae
uv run chabuduo preprocess zurich_tcvae --max-workers 4
uv run chabuduo prepare zurich_tcvae
uv run chabuduo validate zurich_tcvae
```

A processed directory contains:

```text
train.npz
validation.npz
test.npz
metadata.parquet
normalization.json
```

The NPZ `values` arrays are already split, normalized `float32` tensors with
shape `[N, points, features]`. Do not edit generated data files manually.
