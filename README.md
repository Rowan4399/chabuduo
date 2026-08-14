# chabuduo

`chabuduo` is a small command-line research workbench for aircraft trajectory
generation. It prepares fixed datasets, trains one model at a time, preserves
the artifacts, and makes failed generations visible.

The workflow stays explicit:

```text
get → preprocess → prepare → validate → train → sample → plot
```

`get` only makes the raw Parquet source available. `preprocess` loads that file
as a `traffic.core.Traffic` object, works with its individual `Flight` objects,
and writes a selected variable-length Traffic cache. `prepare` converts
that cache into the fixed-shape numeric artifact used by the model. Each
command reads an artifact, writes another artifact, and stops. This keeps
the project understandable to a student and gives coding agents a stable
interface without turning the repository into a workflow framework.

The TCVAE architecture is adapted from the MIT-licensed Code Ocean capsule
[Deep Generative Modelling of Aircraft Trajectories in Terminal Maneuvering
Areas](https://doi.org/10.24433/CO.0357574.V1). See [NOTICE.md](NOTICE.md).

## Current state

The CLI and artifact path work for Beijing and Zurich. The only implemented
model is an unconditional, standard-prior temporal convolutional VAE. It does
**not yet generate convincing unconditional Beijing trajectories**.

Three Beijing representations have been tested or prepared:

| Configuration | Model values | Shape | Status |
|---|---|---:|---|
| `beijing` | projected east, north, MSL altitude | `[N, 200, 3]` | trained for 200 epochs; generations were noisy |
| `beijing_kinematic` | unwrapped track, groundspeed, altitude, elapsed time | `[N, 200, 4]` | trained for 1,000 epochs; best validation epoch 331; prior generations were poor |
| `beijing_kinematic_circular` | sine/cosine track, groundspeed, altitude, elapsed time | `[N, 200, 5]` | trained for 1,000 epochs; best validation epoch 253; sampling and plot review are next |

The 1,000-epoch runs overfit. The best checkpoints are selected by validation
loss; `last.ckpt` is not the recommended checkpoint. More detail, including the
loss values and failed modelling choices, is in
[`lessons_learned.md`](lessons_learned.md).

## Which document to trust

- [`README.md`](README.md): current purpose, supported commands, and status.
- [`AGENTS.md`](AGENTS.md): rules for changing or running the project.
- [`lessons_learned.md`](lessons_learned.md): experiment history and technical
  conclusions.
- [`HANDOFF.md`](HANDOFF.md): student onboarding, reproducibility checklist,
  troubleshooting, and the recommended next experiments.
- [`docs/history/initial_implementation_plan.md`](docs/history/initial_implementation_plan.md):
  superseded first implementation plan.
- [`docs/history/unified_benchmark_vision.md`](docs/history/unified_benchmark_vision.md):
  long-term benchmark ideas, not the current architecture.

## Installation

From this directory:

```bash
uv sync --dev
uv run chabuduo --help
```

`--dev` installs the development group (`pytest`, `ruff`) as well as the runtime
packages. In VS Code, select the interpreter from this project environment:
open the Command Palette, choose **Python: Select Interpreter**, and select
`.venv/bin/python` (macOS/Linux) or `.venv\\Scripts\\python.exe` (Windows).
Then reload the window and verify that the integrated terminal starts in the
project directory. Run commands with `uv run` so the lockfile environment is
used even when VS Code has selected a different interpreter.

PyTorch is pinned to 2.5.1. Linux x86-64 resolves the CUDA 12.1 wheel so remote
training works on the GTX 1080 Ti in `eyjafjallajokull`; macOS uses the normal
platform wheel. The first environment installation can be large.

## First setup and handoff

Before working on the model, follow [`HANDOFF.md`](HANDOFF.md). It explains the
three checks that prevent most wasted experiments: install the dev environment,
verify the Windows-to-Linux GPU SSH connection, and run the local smoke tests.

## Beijing workflow

Beijing can be staged from an existing local Parquet file:

```bash
uv run chabuduo get beijing \
  --source ../2025-07-28.parquet
```

The Feiyou adapter can instead convert the seven configured daily CSV files
into one Traffic-compatible raw Parquet. The source root may contain a nested
directory; the YAML glob finds the daily files but requires exactly one file for
every configured date.

```bash
uv run chabuduo get beijing \
  --config src/chabuduo/configs/beijing_feiyou.yaml \
  --source /path/to/北京20250728-20250803
```

This writes `data/raw/beijing/2025-07-28_2025-08-03.parquet` and a YAML sidecar
with the input hashes, row counts, and unit conversions. Keep the existing
single-day raw, interim, and processed artifacts: the seven-day source is a new
experiment and must use distinct downstream output paths.

The recommended next representation to inspect is circular kinematics:

```bash
uv run chabuduo preprocess beijing_kinematic_circular \
  --source data/raw/beijing/2025-07-28.parquet \
  --max-workers 4

uv run chabuduo prepare beijing_kinematic_circular \
  --selected data/interim/zbaa-landing-40nm.parquet

uv run chabuduo validate \
  data/processed/beijing_kinematic_circular-v1
```

A long local or remote run can be started with:

```bash
uv run chabuduo train \
  data/processed/beijing_kinematic_circular-v1 \
  --config src/chabuduo/configs/tcvae_long.yaml \
  --output runs/beijing-kinematic-circular-v1-tcvae-seed-42-1000 \
  --seed 42
```

That run already exists locally. Its best checkpoint can be sampled and plotted
with:

```bash
uv run chabuduo sample \
  runs/beijing-kinematic-circular-v1-tcvae-seed-42-1000 \
  --count 100 \
  --seed 42

uv run chabuduo plot \
  runs/beijing-kinematic-circular-v1-tcvae-seed-42-1000/samples/seed-42-n-100
```

`--temperature` changes the standard-normal latent sampling scale. It is an
experiment parameter, not a substitute for checking prior/posterior mismatch.

### Shared preprocessing cache

Landing detection, 40 NM cropping, and 200-point resampling are shared by all
Beijing representations. Their reusable preprocessing cache is:

```text
data/interim/zbaa-landing-40nm.parquet
```

Its YAML sidecar records the source hash and selection parameters. Changing the
downstream representation should use this cache instead of repeating landing
detection.

## Zurich workflow

On a cold cache, `get zurich` lets `traffic` download
`landing_zurich_2019` into its own dataset cache. It does **not** copy the file
into `data/raw/`; on a warm cache it simply returns the existing cached path.
`preprocess` resolves that same cache automatically:

```bash
uv run chabuduo get zurich

uv run chabuduo preprocess zurich \
  --max-workers 4
uv run chabuduo prepare zurich \
  --selected data/interim/lszh-40nm-200-points.parquet

uv run chabuduo validate data/processed/zurich-v1

uv run chabuduo train data/processed/zurich-v1 --smoke
```

Zurich currently uses the projected-position representation. Do not compare it
with a Beijing kinematic run as though representation were controlled.

## What `validate` means

`validate` is a read-only training-readiness check, not scientific evaluation.
It checks required files, declared shapes and units, finite and chronological
values, normalization metadata, identifiers, metadata consistency, and split
leakage. It exits with a nonzero status when training should not proceed.

## Artifacts

Prepared datasets contain:

```text
data/processed/<configuration>-<version>/
├── train.npz
├── validation.npz
├── test.npz
├── metadata.parquet
├── normalization.json
├── generation.yaml
└── manifest.yaml
```

The tensor shape and feature count come from `manifest.yaml`; they are not
always `[N, 200, 3]`. The legacy NPZ key `positions` stores normalized model
values even for kinematic representations. Normalization is fitted on the
training split only, with separate statistics for every timestep and feature.

Training runs contain:

```text
runs/<run>/
├── checkpoints/best.ckpt
├── checkpoints/last.ckpt
├── logs/
├── resolved-config.yaml
├── environment.json
├── manifest.yaml
└── status.yaml
```

Sampling writes:

```text
runs/<run>/samples/<sample>/
├── samples.npz
├── generated.parquet
├── manifest.yaml
└── <airport>-trajectories.png  # after `plot`
```

For kinematic runs, `samples.npz` stores model values, sanitized kinematic
values, and reconstructed projected positions. `generated.parquet` is a
`Traffic`-compatible point table. Kinematic landing trajectories are
reconstructed backwards from a Beijing airport anchor, following the endpoint
convention described in the `traffic` generation guide.

## Scope and next decision

The current task is to evaluate the circular-track TCVAE run by comparing:

1. posterior reconstructions;
2. samples from the standard prior;
3. generated track, speed, altitude, and duration distributions;
4. the real/generated trajectory plot.

If prior generations remain poor, the next decision is not “train longer.” The
likely choices are conditioning by runway/flow or changing the latent prior.
DDPM, flow matching, VampPrior, and a full evaluation registry remain future
work.

## Tests

The default tests use synthetic data and perform no dataset download or long
training run:

```bash
uv run pytest
uv run ruff check .
```
