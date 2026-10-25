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

The package contains the TCVAE, the conditional trajectory DDPM, and latent
DDPM training/sampling. The latent DDPM path reuses a frozen TCVAE and its exact
prepared splits; see the commands below. Frozen standard-prior and VampPrior
TCVAE checkpoints can both be loaded. The historical Beijing TCVAE runs do
**not yet generate convincing unconditional Beijing trajectories**.

## Latent DDPM

The supported implementation is `src/chabuduo/models/latent_ddpm.py`, recovered
from the matching local snapshots under
`output/support/conditional-three-models-20261018/remote-source-before/` and
`scripts/archive/latent_denoising_video-v1-seed-42/source/`. Their latent model
sources were identical. The older `.remote_*/scripts/train_latent_ddpm.py`
experiment used posterior means and temporal convolutions; it is not this model.

The fixed design encodes each complete `[200, 4]` trajectory as a 64-dimensional
posterior **sample**, standardizes each latent dimension using training data
only, and trains a residual MLP (width 256, four blocks, timestep embedding 128).
It uses 1,000 linear diffusion steps, noise MSE, AdamW, StepLR, and EMA. The YAML
pins the TCVAE checkpoint epoch, SHA-256, prior, and prepared artifact. No
TCVAE retraining or upstream cache recomputation is required.

Three bundled experiments retain their original parameters:

| Experiment | Frozen TCVAE | Prepared artifact |
|---|---|---|
| `tianjin_noaib_rwy16l_latent_ddpm_vamp_k1000` | VampPrior K1000, epoch 262 | `tianjin_noaib_rwy16l_tcvae_vamp_k1000-v1` |
| `zurich_rwy14_latent_ddpm_vamp_k1000` | VampPrior K1000, epoch 240 | `zurich_rwy14_tcvae_vamp_k1000-v1` |
| `zurich_rwy14_flow162_216_latent_ddpm_mlp` | standard prior, epoch 354 | `zurich_rwy14_flow162_216_tcvae-v1` |

Each requires the referenced files to exist locally. On this checkout, the
Tianjin prepared artifact and both trained checkpoints are available. The
default `.venv` lacks PyTorch; the existing `.venv-video` CPU environment now
has the package and test dependencies installed. Select it in PowerShell:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$PWD/.venv-video"
$env:UV_CACHE_DIR = "$PWD/.uv-task-cache"
$env:TRAFFIC_CACHE_PATH = "$PWD/data/cache/traffic"
```

The local Traffic cache contains airport and runway metadata. Use a new output
directory when testing, because existing runs and samples are preserved:

```powershell
uv run --no-sync chabuduo prepare tianjin_noaib_rwy16l_latent_ddpm_vamp_k1000
uv run --no-sync chabuduo validate tianjin_noaib_rwy16l_latent_ddpm_vamp_k1000
uv run --no-sync chabuduo train tianjin_noaib_rwy16l_latent_ddpm_vamp_k1000 --smoke --output runs/latent-ddpm-local-smoke
uv run --no-sync chabuduo sample tianjin_noaib_rwy16l_latent_ddpm_vamp_k1000 --run runs/latent-ddpm-local-smoke --output runs/latent-ddpm-local-smoke/samples/review
uv run --no-sync chabuduo plot tianjin_noaib_rwy16l_latent_ddpm_vamp_k1000 --sample runs/latent-ddpm-local-smoke/samples/review
uv run --no-sync chabuduo plot-loss runs/latent-ddpm-local-smoke
```

For latent DDPM, `prepare` checks and returns the referenced TCVAE artifact. It
does not split, normalize, or encode it again. `train` creates
`artifacts/latent_dataset.npz`, `latent_normalization.json`, and
`latent_manifest.json`, then writes `checkpoints/best.pt`, `last.pt`, CSV metrics,
the environment, the consumed configuration, and a run summary. `--smoke` uses
up to eight trajectories per split, a smaller MLP, and one CPU training batch
and validation batch. Its outputs only check the implementation path.

Resume the same smoke run without encoding the latent cache again:

```powershell
uv run --no-sync chabuduo train tianjin_noaib_rwy16l_latent_ddpm_vamp_k1000 --smoke --output runs/latent-ddpm-local-smoke --resume --additional-epochs 1
```

For a formal run, commit its YAML first, choose a fresh `--output`, and omit
`--smoke` to use the declared 500-epoch budget. Always resume with the same
smoke/formal mode. `--learning-rate` optionally overrides the optimizer rate
when resuming. Previous checkpoints are retained under `resume_sources/`.
Sampling selects `best.pt`, uses its EMA weights when present, decodes through
the pinned TCVAE, and retains both raw and standardized generated latent vectors
in `samples.npz`. Geographic reconstruction anchors at the configured runway
threshold; `sample_summary.json` records clipping and elapsed-time repair;
this does not establish physical realism.

## Historical Beijing experiments

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

## Code and run directories

`src/chabuduo/` contains the supported package implementation. `scripts/`
contains thin command-line entry points and exploratory ideas, including
evaluation calculations that are not ready to become package code.

`runs/` stores model training information and its outputs. Evaluation scripts
read those artifacts from `scripts/`; do not copy executable scripts into each
run. Existing evaluation data and figures keep their paths during this move.
Historical source versions are preserved under `scripts/archive/`, with the
old-to-new paths listed in `scripts/MOVED_FROM_RUNS.md`.

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
Flow matching and a full evaluation registry remain future work. Frozen
VampPrior checkpoints are supported by the latent DDPM path above.

## Tests

The default tests use synthetic data and perform no dataset download or long
training run:

```bash
uv run pytest
uv run ruff check .
```
