# Working in `chabuduo`

These instructions apply to the whole project. Keep the project small, readable,
and usable one command at a time.

## Sources of truth

- Read `README.md` for current status and the next experiment.
- Read `lessons_learned.md` before repeating or extending a failed experiment.
- Treat files under `docs/history/` as historical context only. They are not the
  current architecture or task list.
- When documentation disagrees, prefer `README.md`, then this file, then the
  committed experiment YAML used by every CLI stage.

## Preserve current work

- Read the latest version of a file immediately before editing it.
- Keep changes small and targeted. Do not rewrite nearby code, formatting, or
  comments unless the task requires it.
- Never replace the user's edits with an earlier agent-generated version.
- Do not edit files when the user asks only for analysis.
- Leave unrelated modified and untracked files alone.
- Never delete data, checkpoints, samples, plots, or caches requested by the
  user. Ask before removing a large artifact.
- State clearly when a run occurs under `/tmp` or on another machine. A remote
  run is not delivered until its outputs have been copied back into this
  project.

## Follow the visible workflow

The CLI is the supported project interface:

```text
get → preprocess → validate → train → sample → plot
```

- Prefer running a `chabuduo` command over reaching into internal modules.
- Each command should read an artifact, write another artifact, and stop.
- Keep commands noninteractive and make paths and seeds explicit.
- Do not add an orchestration framework or an automatic “run everything” command
  without a concrete need.
- Preserve the manual stages: they make failures understandable to students and
  give coding agents a stable interface.
- Print the output path at the end of every successful command.

## Keep the architecture modest

- Prefer ordinary functions and a few clear modules over registries, factories,
  plugin systems, and inheritance trees.
- Do not add cache databases, pipeline-version classes, notebook detection,
  generic utility modules, or defensive wrappers without a measured problem.
- A YAML file should contain parameters, not executable workflow logic.
- Files are valid interfaces between stages. A clear Parquet, NPZ, YAML, or
  checkpoint is often enough.
- Start with one complete vertical path before adding another model family.
- Move exploratory code into the package only when it is needed to reproduce a
  result. `# %%` scripts remain appropriate for exploration.

## Use caches deliberately

- Cache expensive shared computations such as landing detection, terminal-area
  selection, and fixed-point resampling.
- Do not confuse a representation-specific processed artifact with a reusable
  selected-trajectory cache.
- Cache identity should cover the source and the parameters that change the
  result. Keep the identity small and readable.
- Never recompute an expensive stage merely because a downstream representation
  changed.
- Test both a cold computation and a warm cache load when changing cache code.
- A fixed, descriptive cache filename is preferable to an opaque artifact
  manager.

## Use `traffic` as the aviation-domain layer

Make a local checkout and do edits there if necessary.
Source is https://github.com/xoolive/traffic, prefer the `prepare-v3` branch

- Prefer the direct, documented `traffic` API over custom trajectory
  abstractions.
- Use canonical expressions such as `airports["ZBAA"]`.
- If a documented public API fails, investigate it and add a focused test rather
  than teaching a workaround.
- Keep `Traffic`, `Flight`, `LazyTraffic`, and the underlying `.data` DataFrame
  distinct in code and explanations.
- Name variables and parameters holding a `Traffic` object `t`, never `traffic`.
- Use `LazyTraffic.eval(cache_file=...)` for expensive flight-wise operations
  when that API fits the stage.
- `Flight.distance(Airport)` returns nautical miles.
- Load only needed source columns when aliases such as `lat/lng` and
  `latitude/longitude` would otherwise create duplicate columns.
- Do not make neural model classes depend directly on `Traffic`. Convert at the
  dataset and sampling boundaries.

## Be explicit about representations and units

- Record feature order, units, sequence length, direction, projection, and
  endpoint convention in every prepared artifact.
- `traffic` altitude is conventionally in feet and groundspeed in knots.
- Projected coordinates are in metres; elapsed time is in seconds.
- Do not call projected east/north plus MSL altitude true three-dimensional ENU.
- Treat track as a circular variable. If using raw degrees, explain unwrapping
  and branch conventions. Prefer sine/cosine when scalar angle discontinuities
  damage the model.
- Fit normalization on the training split only.
- For fixed-length generation, use per-timestep/per-feature training statistics
  unless an experiment explicitly tests another normalization.
- Split accepted trajectories before fitting normalization, clustering, labels,
  or any other learned preprocessing.

## Train and evaluate honestly

- Run `validate` before training.
- Start new code paths with `--smoke`, then use the declared training budget.
- Commit the experiment YAML before training, and store the environment, logs,
  best checkpoint, and last checkpoint.
- Select checkpoints with validation data, never test data.
- Inspect training loss, validation loss, reconstruction loss, and KL loss. Do
  not report only the last epoch.
- More epochs are not automatically better. Check where validation loss is
  minimized and look for overfitting.
- A negative Gaussian log-likelihood is possible. Compare values only under the
  same loss definition.
- Reconstruction quality and prior-sample quality are different questions.
  Check both.
- Generate and plot samples before declaring a model successful.
- Keep failed or poor plots when they explain the next experiment.
- Use identical trajectories, splits, normalization, seeds, sample counts, and
  evaluation inputs when comparing model families.

## GPU runs

- Check GPU availability, driver version, free disk space, and PyTorch CUDA
  compatibility before training.
- If not available locally, suggest to run it remotely over ssh.
- The GTX 1080 Ti requires a compatible CUDA/PyTorch build. Linux x86-64 uses
  the PyTorch 2.5.1 CUDA 12.1 wheel; do not force that wheel on macOS.
- Redirect long training output to a log file and inspect the final status and
  metrics.
- Copy the complete run back under `chabuduo/runs/`, including checkpoints,
  CSV metrics, environment, samples, plots, and remote log.
- Do not leave the only copy of a result on the remote host.

## Plot geographic data correctly

- Follow examples in the `traffic` documentation.
- Use projections from `cartes.crs` and let `traffic` objects plot themselves
  where practical.
- Never draw longitude and latitude as ordinary Cartesian x/y coordinates.
- Pass `PlateCarree()` as the source transform for manually generated
  longitude/latitude geometry.
- Avoid accidental network calls in plots. For example, use
  `airport.plot(ax, footprint=False)` when an OpenStreetMap footprint is not
  needed.
- Compare generated trajectories with held-out real trajectories using the same
  extent and projection.

## Tooling and validation

- Run Python only through `uv`; do not invoke global `python` or `python3`.
- Use `uv run --no-sync` when the existing environment is already correct.
- Run Ruff on changed Python files.
- Run focused tests first, then the full test suite when shared contracts change.
- Tests must not download Zurich data, read the full Beijing source, or perform a
  long training run by default.
- Keep optional visualization work separate from the core data path where
  practical.
- State blockers and uncertainty plainly instead of hiding them behind more
  code.

## Write for students and researchers

- Use short, natural comments that explain why a step exists.
- Do not narrate obvious Python syntax.
- Avoid grand claims and framework language. Give shapes, units, counts, epochs,
  and file paths.
- Explain the limits of generated data. Endpoint anchoring, clipping, or a good
  reconstruction loss does not prove physical realism.
