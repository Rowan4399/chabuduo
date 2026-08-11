# Local data

This directory is created and filled by the manual project workflow. The files
are local experiment inputs and prepared artifacts; they are intentionally
ignored by Git because the Beijing source and derived arrays can be large.

## Layout

```text
data/
├── raw/
│   └── beijing/       # staged local Beijing Parquet source
├── interim/           # selected/resampled Traffic Parquet caches
└── processed/         # immutable generation artifacts for model training
```

Stage an existing Beijing Parquet, or convert the configured week of Feiyou
CSV files:

```bash
uv run chabuduo get beijing --source ../2025-07-28.parquet
uv run chabuduo get beijing \
  --config src/chabuduo/configs/beijing_feiyou.yaml \
  --source /path/to/北京20250728-20250803
```

The second command writes
`data/raw/beijing/2025-07-28_2025-08-03.parquet`. Use a period-specific interim
cache and processed artifact for that source rather than replacing the existing
single-day experiment.

Continue the current single-day workflow with:

```bash
uv run chabuduo preprocess beijing_kinematic_circular \
  --source data/raw/beijing/2025-07-28.parquet --max-workers 4
uv run chabuduo prepare-generation beijing_kinematic_circular \
  --selected data/interim/zbaa-40nm-200-points.parquet
uv run chabuduo validate data/processed/beijing_kinematic_circular-v1
```

Generation artifacts contain `train.npz`, `validation.npz`, `test.npz`,
`metadata.parquet`, `normalization.json`, `generation.yaml`, and `manifest.yaml`.
They are immutable: create a new output directory for a changed experiment.

Do not delete raw sources, caches, or prepared artifacts without checking
whether an existing run depends on them. The manifest records source hashes and
representation details needed to interpret a run.
