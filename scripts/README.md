# Scripts

The supported implementation belongs in `src/chabuduo/`. This directory holds
thin CLI entry points and exploratory work that is still taking shape. An
evaluation calculation can remain an ordinary script here until it is needed
as a stable, reproducible package feature.

Run scripts from the project root through `uv`. Keep inputs, output paths, and
seeds explicit. A script should read artifacts, perform its calculation, write
the result, print the output path, and stop.

## Shared entry points

- `plot_samples.py` handles exploratory trajectory plots.
- `plot_loss.py` handles exploratory training loss plots.
- The `chabuduo` CLI remains the interface for get, preprocess, prepare,
  validate, train, sample, and plot.
- Latent DDPM training and sampling live in
  `src/chabuduo/models/latent_ddpm.py`; use `chabuduo train ... --smoke`,
  `train ... --resume`, `sample`, `plot`, and `plot-loss` as documented in the
  root README. No training implementation is copied into `scripts/`.
- `latent_denoising_video.py` only records sampler states and renders videos.
  It imports the package model and records the hashes of the source it uses.
- `.remote_*/scripts/train_latent_ddpm.py` is an older posterior-mean/convolution
  experiment. Keep it as historical context; it is not the residual MLP trainer.

## Evaluation tools moved from runs

| Script | Calculation |
|---|---|
| `inspect_checkpoint.py` | Extract the resolved TCVAE configuration from a checkpoint; accepts checkpoint and output as positional paths. |
| `component_diagnostics.py` | Inspect VampPrior components and training posteriors; accepts `--checkpoint`, `--train`, `--output`, and `--seed`. |
| `prepare_simulator_reference.py` | Prepare reference inputs for the simulator comparison; see `--help`. |
| `audit_gaps.py` | Audit Tianjin position/time gaps; see `--help`. |
| `summarize_runways.py` | Summarize the selected Tianjin runway data; see `--help`. |
| `audit_epoch500.py` | Compare the two saved Zurich DDPM epoch-500 configurations; its original run and report paths are retained. |
| `verify_review.py` | Verify the frozen RWY14 evaluation artifacts. |
| `verify_training_compare.py` | Verify the frozen training-reference display and package its results. |
| `evaluate_generation_distributions.py` | Calculate distributions from fixed inputs; accepts `--inputs`, `--output`, `--seed`, and `--smoke`. |
| `replot_generation_distributions.py` | Replot existing metrics; accepts `--evaluation`, `--inputs`, and a new `--output`. |

`audit_epoch500.py`, `verify_review.py`, and `verify_training_compare.py` still
use their original fixed experiment directories.
They are exploratory calculations, not general package commands. Moving them
does not retrain, resample, recalibrate, or regenerate existing results.

## Historical sources

`archive/` preserves old source versions and completed one-time patches.
These are historical references rather than runnable current entry points.
`MOVED_FROM_RUNS.md` records their previous and current locations. Existing
training manifests and scientific results retain their original provenance.

Use the current script directly when running another analysis. Record its path
and hash in the result metadata instead of copying another executable into
`runs/`. Model training information belongs in `runs/`; evaluation code belongs
here. Existing evaluation outputs keep their paths in this first cleanup.

Distribution replotting locates the recorded calculation version by the source
hash in `manifest.json`, using the current script or a distinct historical
version in `archive/`. New evaluation and replot outputs contain results and
source identities rather than executable script copies. Reproduction commands
are written in Markdown.
