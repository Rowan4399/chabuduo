# chabuduo

`chabuduo` is a small command-line research workbench for aircraft trajectory
generation. The supported workflow is deliberately explicit:

```text
get → preprocess → prepare → validate → train → sample → plot
```

Every command reads one artifact, writes another artifact, prints its output
path, and stops. Experiment parameters live in YAML; checkpoints and run
manifests record what actually ran.

The TCVAE architecture is adapted from the MIT-licensed Code Ocean capsule
[Deep Generative Modelling of Aircraft Trajectories in Terminal Maneuvering
Areas](https://doi.org/10.24433/CO.0357574.V1). See `NOTICE.md`.

## Current models

- TCVAE with a standard normal prior. VampPrior is the next model change and is
  not part of the current baseline.
- Direct trajectory DDPM. The current Zurich baseline is an unconditional,
  full-resolution temporal dilated residual network trained to predict noise.
- The older conditional dual-branch DDPM remains available for its historical
  configuration, but is not the current Zurich baseline.

The current comparable Zurich experiments use RWY 14, initial flow `162-216`,
200 chronological points, and the same four features: unwrapped track in
degrees, groundspeed in knots, altitude in feet, and elapsed time in seconds.
Both reconstruct landings backwards from the runway threshold.

## Installation

```bash
uv sync --dev
uv run chabuduo --help
```

Run Python commands through `uv`. The remote Windows environment uses the
PyTorch CUDA wheel selected in `pyproject.toml`.

## Experiments

One YAML describes each complete experiment. The two current Zurich baseline
files are:

```text
src/chabuduo/configs/zurich_rwy14_flow162_216_ddpm_unconditional.yaml
src/chabuduo/configs/zurich_rwy14_flow162_216_tcvae.yaml
```

Older Beijing and Zurich YAML files remain for reproducibility. A continuation
does not need a second `_resume700` or `_one_epoch` YAML; use the original
experiment configuration with the resume options below.

## Running one experiment

Replace `<experiment>` with the YAML filename without `.yaml`:

```bash
uv run chabuduo get <experiment>
uv run chabuduo preprocess <experiment> --max-workers 4
uv run chabuduo prepare <experiment>
uv run chabuduo validate <experiment>
uv run chabuduo train <experiment> --smoke
uv run chabuduo train <experiment>
uv run chabuduo sample <experiment>
uv run chabuduo plot <experiment>
```

`validate` checks readiness; it is not a scientific evaluation. The smoke run
uses one small CPU epoch and a separate `-smoke` run directory.

## Resuming training

Both TCVAE and DDPM resume from the run's `last` checkpoint, including optimizer
and scheduler state. The historical best checkpoint remains the sampling
checkpoint unless the resumed validation score improves it.

```bash
uv run chabuduo train <experiment> --resume --additional-epochs 100
```

To intentionally change the restored optimizer learning rate:

```bash
uv run chabuduo train <experiment> --resume --additional-epochs 100 \
  --learning-rate 0.00005
```

Before training continues, the command copies the previous best checkpoint,
last checkpoint, and run summary into `resume_sources/epoch-<n>/`. Each resume
event is appended to `run_summary.json`.

## Plotting

The package plot stage and the scripts use the same plotting implementation.
The trajectory figure always shows held-out trajectories on the left and up to
100 generated trajectories on the right, with the same projection and limits.

```bash
uv run chabuduo plot <experiment> --count 100
uv run scripts/plot_samples.py <experiment> --run runs/<run-directory>
```

Loss plotting accepts either the DDPM CSV or all Lightning CSV versions. Its
left panel shows the complete history and its right panel starts after warm-up:

```bash
uv run scripts/plot_loss.py runs/<run-directory> --warmup-epochs 20
```

TCVAE latent-space diagnostics stay model-specific and outside the package:

```bash
uv run scripts/analyze_tcvae_latent.py --run runs/<tcvae-run> \
  --artifact data/processed/<tcvae-artifact>
```

## Artifact interfaces

Prepared datasets contain ready tensors and numerical metadata:

```text
data/processed/<experiment>-<version>/
├── train.npz
├── validation.npz
├── test.npz
├── metadata.parquet
└── normalization.json
```

Each NPZ records normalized values, elapsed time, trajectory identifiers, and
split-group identifiers. Feature order, units, direction, sequence length,
anchor, and split policy come from the experiment YAML. Normalization is fitted
on the training split only and stores one mean and standard deviation for each
timestep and feature.

Training runs use one common layout. TCVAE checkpoint names end in `.ckpt` and
DDPM checkpoint names end in `.pt`:

```text
runs/<experiment>-<version>-seed-<seed>/
├── checkpoints/best.<ckpt|pt>
├── checkpoints/last.<ckpt|pt>
├── logs/
├── environment.json
├── run_summary.json
├── resume_sources/                 # present after a resume
├── plots/
└── samples/
```

`run_summary.json` records the exact prepared-artifact identity, monitored
metric, best and last checkpoints, hashes, final learning rate, completion
state, and resume history.

Each sampling directory contains:

```text
samples/<sampling-identity>/
├── samples.npz
├── generated.parquet
├── sample_summary.json
└── trajectory_comparison.png       # after plot
```

`samples.npz` stores inverse-normalized model values in YAML feature order plus
elapsed time. `generated.parquet` stores the reconstructed point table with
canonical track, speed, altitude, timestamps, projected `x/y`, and model
features. `sample_summary.json` identifies the run, prepared artifact,
checkpoint hash, seed, count, feature order, and sequence length.

## Tests

The default tests use synthetic data and do not download datasets or start a
long training run:

```bash
uv run pytest
uv run ruff check .
```
