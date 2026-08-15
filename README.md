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

Three historical Beijing runs remain useful evidence. Their directory names
predate the current single-config workflow:

| Historical run | Model values | Shape | Status |
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

## One YAML per experiment

The committed YAML is the only configuration source of truth. Current choices
are:

```text
src/chabuduo/configs/beijing_tcvae.yaml
src/chabuduo/configs/beijing_ddpm.yaml
src/chabuduo/configs/zurich_tcvae.yaml
src/chabuduo/configs/zurich_ddpm.yaml
```

Each file contains source, preprocessing, tensor preparation, model, optimizer,
training, and sampling parameters. Every command receives the same experiment
name.

## Beijing workflow

The Beijing configurations convert both configured Feiyou weeks. Pass the
parent directory containing the two dated source directories:

```bash
uv run chabuduo get beijing_tcvae --source /path/to/feiyou-root
uv run chabuduo preprocess beijing_tcvae --max-workers 4
uv run chabuduo prepare beijing_tcvae
uv run chabuduo validate beijing_tcvae
uv run chabuduo train beijing_tcvae --smoke
```

After the smoke run succeeds, omit `--smoke` to use the training budget declared
in `beijing_tcvae.yaml`. Sampling and plotting also read their count, seed, and
model parameters from that file:

```bash
uv run chabuduo sample beijing_tcvae
uv run chabuduo plot beijing_tcvae
```

The Beijing TCVAE uses circular track features. The Beijing DDPM configuration
uses scalar unwrapped track and records the conditional-DDPM design, but DDPM
training and flow labels are not implemented yet.

### Shared preprocessing cache

Landing detection and 40 NM selection are shared by the Beijing experiments:

```text
data/interim/zbaa-landing-40nm-2025-2weeks-v1.parquet
```

The cache contains variable-length selected `Traffic`. Fixed-length resampling
happens in `prepare`. There is no generated YAML sidecar; when the raw source or
selection policy changes, choose a new `--output` path or move the old cache
aside before recomputing it.

## Zurich workflow

On a cold cache, `get` lets `traffic` download `landing_zurich_2019` into its
own dataset cache. It does not copy that source into `data/raw/`:

```bash
uv run chabuduo get zurich_tcvae
uv run chabuduo preprocess zurich_tcvae --max-workers 4
uv run chabuduo prepare zurich_tcvae
uv run chabuduo validate zurich_tcvae
uv run chabuduo train zurich_tcvae --smoke
uv run chabuduo sample zurich_tcvae
uv run chabuduo plot zurich_tcvae
```

Zurich TCVAE and DDPM configurations use the same four kinematic values. The
DDPM configuration remains non-executable until its trainer and conditioning
labels are added.

## What `validate` means

`validate` is a read-only training-readiness check, not scientific evaluation.
It checks required files, declared shapes and units, finite and chronological
values, normalization metadata, identifiers, metadata consistency, and split
leakage. It exits with a nonzero status when training should not proceed.

## Artifacts

Prepared datasets contain only ready tensors and their numerical metadata:

```text
data/processed/<experiment>-<version>/
├── train.npz
├── validation.npz
├── test.npz
├── metadata.parquet
└── normalization.json
```

Each NPZ stores normalized `values` with shape `[N, points, features]`, elapsed
time, and trajectory/split identifiers. Feature order, units, and point count
come from the original experiment YAML. Normalization is fitted on the training
split only, with separate statistics for every timestep and feature.

Training runs contain:

```text
runs/<experiment>-<version>-seed-<seed>/
├── checkpoints/best.ckpt
├── checkpoints/last.ckpt
├── logs/
├── environment.json
└── samples/
```

Sampling writes `samples.npz`, `generated.parquet`, and—after `plot`—the PNG.
No stage generates another YAML file.

`samples.npz` stores inverse-normalized model `values` in the feature order from
the experiment YAML plus the elapsed-time context used for reconstruction.
`generated.parquet` stores the reconstructed `Traffic` point table, including
canonical track/speed/altitude/timestamps, projected `x/y`, and the original
model columns. Kinematic landings are reconstructed backwards from the airport
anchor following the `traffic` generation guide.

## Scope and next decision

The current task is to evaluate the circular-track TCVAE run by comparing:

1. posterior reconstructions;
2. samples from the standard prior;
3. generated track, speed, altitude, and duration distributions;
4. the real/generated trajectory plot.

If prior generations remain poor, the next decision is not “train longer.” The
likely choices are conditioning by runway/flow or changing the latent prior.
Conditional DDPM training and sampling are available, but preparation still
needs the `model_condition_id` flow labels. DDPM sampling balances the requested
total sample count across the condition IDs present in the training split.
Flow matching, VampPrior, and a full evaluation registry remain future work.

## Tests

The default tests use synthetic data and perform no dataset download or long
training run:

```bash
uv run pytest
uv run ruff check .
```
