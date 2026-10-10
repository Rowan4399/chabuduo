# Evaluation code cleanup

Organized on 2026-10-11. Active evaluation tools are directly in `scripts/`. Distinct historical sources remain in `scripts/archive/`. Exact duplicates, completed plotting-patch helpers, and obsolete launch commands were removed. Model files, checkpoints, caches, metrics, figures, and evaluation results were kept. Historical manifests retain their original execution paths and hashes.

| Previous location | Current source or disposition |
|---|---|
| `runs/distribution-evaluation-20261008-smoke-v1/evaluate_generation_distributions.py` | `scripts/archive/distribution-evaluation-20261008-smoke-v1/evaluate_generation_distributions.py` |
| `runs/distribution-evaluation-20261008-smoke-v3/evaluate_generation_distributions.py` | `scripts/archive/distribution-evaluation-20261008-smoke-v3/evaluate_generation_distributions.py` |
| `runs/distribution-evaluation-20261008-v1/evaluate_generation_distributions.py` | `scripts/archive/distribution-evaluation-20261008-v1/evaluate_generation_distributions.py` |
| `runs/distribution-evaluation-20261008-v2/evaluate_generation_distributions.py` | `scripts/archive/distribution-evaluation-20261008-v2/evaluate_generation_distributions.py` |
| `runs/distribution-evaluation-20261008-v3/evaluate_generation_distributions.py` | `scripts/evaluate_generation_distributions.py` (identical copy retained) |
| `runs/distribution-evaluation-20261008-v3/reproduce.ps1` | `scripts/archive/distribution-evaluation-20261008-v3/reproduce.ps1` |
| `runs/distribution-evaluation-20261008-v3/verify_report.py` | `scripts/verify_report.py` |
| `runs/distribution-evaluation-20261008-v4/evaluate_generation_distributions.py` | `scripts/evaluate_generation_distributions.py` (identical copy retained) |
| `runs/distribution-evaluation-20261008-v4/replot_generation_distributions.py` | `scripts/archive/distribution-evaluation-20261008-v4/replot_generation_distributions.py` |
| `runs/distribution-evaluation-20261008-v4/reproduce.ps1` | `scripts/archive/distribution-evaluation-20261008-v4/reproduce.ps1` |
| `runs/distribution-evaluation-20261008-v4/verify_report.py` | `scripts/verify_report.py` (identical copy retained) |
| `runs/distribution-evaluation-20261008-v4-reviewed/evaluate_generation_distributions.py` | `scripts/evaluate_generation_distributions.py` (identical copy retained) |
| `runs/distribution-evaluation-20261008-v4-reviewed/replot_generation_distributions.py` | `scripts/replot_generation_distributions.py` (identical copy retained) |
| `runs/distribution-evaluation-20261008-v4-reviewed/reproduce.ps1` | `scripts/archive/distribution-evaluation-20261008-v4-reviewed/reproduce.ps1` |
| `runs/distribution-evaluation-20261008-v4-reviewed/verify_report.py` | `scripts/verify_report.py` (identical copy retained) |
| `runs/distribution-evaluation-20261008-v4-smoke/evaluate_generation_distributions.py` | `scripts/evaluate_generation_distributions.py` (identical copy retained) |
| `runs/distribution-evaluation-20261008-v4-smoke/replot_generation_distributions.py` | `scripts/archive/distribution-evaluation-20261008-v4-smoke/replot_generation_distributions.py` |
| `runs/distribution-evaluation-20261008-v4-smoke/reproduce.ps1` | `scripts/archive/distribution-evaluation-20261008-v3/reproduce.ps1` (identical copy retained) |
| `runs/distribution-evaluation-20261008-v4-smoke/verify_report.py` | `scripts/verify_report.py` (identical copy retained) |
| `runs/latent-space-review-20261009/global-standard-prior/inputs/global-standard-prior-export/export_posterior_latents.py` | `scripts/export_posterior_latents.py` (identical copy retained) |
| `runs/latent-space-review-20261009/global-standard-prior/inspect_checkpoint.py` | `scripts/inspect_checkpoint.py` |
| `runs/latent_denoising_video-v1-seed-42/review_outputs.py` | `scripts/review_outputs.py` |
| `runs/latent_denoising_video-v1-seed-42/source/src/chabuduo/__init__.py` | `src/chabuduo/__init__.py` (identical copy retained) |
| `runs/latent_denoising_video-v1-seed-42/source/src/chabuduo/artifacts.py` | `scripts/archive/latent_denoising_video-v1-seed-42/source/src/chabuduo/artifacts.py` |
| `runs/latent_denoising_video-v1-seed-42/source/src/chabuduo/cli.py` | `scripts/archive/latent_denoising_video-v1-seed-42/source/src/chabuduo/cli.py` |
| `runs/latent_denoising_video-v1-seed-42/source/src/chabuduo/config.py` | `scripts/archive/latent_denoising_video-v1-seed-42/source/src/chabuduo/config.py` |
| `runs/latent_denoising_video-v1-seed-42/source/src/chabuduo/get.py` | `src/chabuduo/get.py` (identical copy retained) |
| `runs/latent_denoising_video-v1-seed-42/source/src/chabuduo/models/__init__.py` | `scripts/archive/latent_denoising_video-v1-seed-42/source/src/chabuduo/models/__init__.py` |
| `runs/latent_denoising_video-v1-seed-42/source/src/chabuduo/models/ddpm.py` | `scripts/archive/latent_denoising_video-v1-seed-42/source/src/chabuduo/models/ddpm.py` |
| `runs/latent_denoising_video-v1-seed-42/source/src/chabuduo/models/latent_ddpm.py` | `scripts/archive/latent_denoising_video-v1-seed-42/source/src/chabuduo/models/latent_ddpm.py` |
| `runs/latent_denoising_video-v1-seed-42/source/src/chabuduo/models/tcvae.py` | `scripts/archive/latent_denoising_video-v1-seed-42/source/src/chabuduo/models/tcvae.py` |
| `runs/latent_denoising_video-v1-seed-42/source/src/chabuduo/plotting.py` | `scripts/archive/latent_denoising_video-v1-seed-42/source/src/chabuduo/plotting.py` |
| `runs/latent_denoising_video-v1-seed-42/source/src/chabuduo/prepare.py` | `scripts/archive/latent_denoising_video-v1-seed-42/source/src/chabuduo/prepare.py` |
| `runs/latent_denoising_video-v1-seed-42/source/src/chabuduo/preprocess.py` | `scripts/archive/latent_denoising_video-v1-seed-42/source/src/chabuduo/preprocess.py` |
| `runs/latent_denoising_video-v1-seed-42/source/src/chabuduo/sampling.py` | `scripts/archive/latent_denoising_video-v1-seed-42/source/src/chabuduo/sampling.py` |
| `runs/latent_denoising_video-v1-seed-42/source/src/chabuduo/sources/__init__.py` | `src/chabuduo/sources/__init__.py` (identical copy retained) |
| `runs/latent_denoising_video-v1-seed-42/source/src/chabuduo/sources/feiyou.py` | `scripts/archive/latent_denoising_video-v1-seed-42/source/src/chabuduo/sources/feiyou.py` |
| `runs/latent_denoising_video-v1-seed-42/source/src/chabuduo/train.py` | `scripts/archive/latent_denoising_video-v1-seed-42/source/src/chabuduo/train.py` |
| `runs/latent_denoising_video-v1-seed-42/source/src/chabuduo/validate.py` | `scripts/archive/latent_denoising_video-v1-seed-42/source/src/chabuduo/validate.py` |
| `runs/plot-lonlat-20261010/add_cache_inputs.py` | Removed: completed one-time patch |
| `runs/plot-lonlat-20261010/apply_remote_patch.py` | Removed: completed one-time patch |
| `runs/plot-lonlat-20261010/finalize_result.py` | Removed: completed one-time patch |
| `runs/plot-lonlat-20261010/fix_remote_format.py` | Removed: completed one-time patch |
| `runs/plot-lonlat-20261010/remote-plot-lonlat.patch` | `scripts/archive/plot-lonlat-20261010/remote-plot-lonlat.patch` |
| `runs/plot-lonlat-20261010/source-after/scripts/build_review_bundle.py` | `scripts/archive/plot-lonlat-20261010/source-after/scripts/build_review_bundle.py` |
| `runs/plot-lonlat-20261010/source-after/scripts/compare_three_latent_sources.py` | `scripts/archive/plot-lonlat-20261010/source-after/scripts/compare_three_latent_sources.py` |
| `runs/plot-lonlat-20261010/source-after/scripts/plot_ddpm_diffusion_stages.py` | `scripts/archive/plot-lonlat-20261010/source-after/scripts/plot_ddpm_diffusion_stages.py` |
| `runs/plot-lonlat-20261010/source-after/scripts/plot_tcvae_reconstructions.py` | `scripts/archive/plot-lonlat-20261010/source-after/scripts/plot_tcvae_reconstructions.py` |
| `runs/plot-lonlat-20261010/source-after/scripts/refine_ddpm_denoising.py` | `scripts/archive/plot-lonlat-20261010/source-after/scripts/refine_ddpm_denoising.py` |
| `runs/plot-lonlat-20261010/source-after/scripts/trajectory_rules.html` | `scripts/archive/plot-lonlat-20261010/source-after/scripts/trajectory_rules.html` |
| `runs/plot-lonlat-20261010/source-after/scripts/trajectory_rules.py` | `scripts/archive/plot-lonlat-20261010/source-after/scripts/trajectory_rules.py` |
| `runs/plot-lonlat-20261010/source-after/src/chabuduo/cli.py` | `scripts/archive/plot-lonlat-20261010/source-after/src/chabuduo/cli.py` |
| `runs/plot-lonlat-20261010/source-after/src/chabuduo/plotting.py` | `scripts/archive/plot-lonlat-20261010/source-after/src/chabuduo/plotting.py` |
| `runs/plot-lonlat-20261010/source-after/tests/test_plotting_lonlat.py` | `scripts/archive/plot-lonlat-20261010/source-after/tests/test_plotting_lonlat.py` |
| `runs/plot-lonlat-20261010/source-before/scripts/build_review_bundle.py` | `scripts/build_review_bundle.py` (identical copy retained) |
| `runs/plot-lonlat-20261010/source-before/scripts/compare_three_latent_sources.py` | `scripts/archive/plot-lonlat-20261010/source-before/scripts/compare_three_latent_sources.py` |
| `runs/plot-lonlat-20261010/source-before/scripts/plot_ddpm_diffusion_stages.py` | `scripts/plot_ddpm_diffusion_stages.py` (identical copy retained) |
| `runs/plot-lonlat-20261010/source-before/scripts/plot_tcvae_reconstructions.py` | `scripts/archive/plot-lonlat-20261010/source-before/scripts/plot_tcvae_reconstructions.py` |
| `runs/plot-lonlat-20261010/source-before/scripts/refine_ddpm_denoising.py` | `scripts/archive/plot-lonlat-20261010/source-before/scripts/refine_ddpm_denoising.py` |
| `runs/plot-lonlat-20261010/source-before/scripts/trajectory_rules.html` | `scripts/archive/plot-lonlat-20261010/source-before/scripts/trajectory_rules.html` |
| `runs/plot-lonlat-20261010/source-before/scripts/trajectory_rules.py` | `scripts/trajectory_rules.py` (identical copy retained) |
| `runs/plot-lonlat-20261010/source-before/src/chabuduo/cli.py` | `scripts/archive/latent_denoising_video-v1-seed-42/source/src/chabuduo/cli.py` (identical copy retained) |
| `runs/plot-lonlat-20261010/source-before/src/chabuduo/plotting.py` | `scripts/archive/latent_denoising_video-v1-seed-42/source/src/chabuduo/plotting.py` (identical copy retained) |
| `runs/rwy14-review-v1/source/finalize_tcvae_selection.py` | `scripts/archive/rwy14-review-v1/source/finalize_tcvae_selection.py` |
| `runs/rwy14-review-v1/source/inspect_inputs.py` | `scripts/inspect_inputs.py` |
| `runs/rwy14-review-v1/source/package_correction.py` | `scripts/archive/rwy14-review-v1/source/package_correction.py` |
| `runs/rwy14-review-v1/source/page-check.js` | `scripts/archive/rwy14-review-v1/source/page-check.js` |
| `runs/rwy14-review-v1/source/replace_tcvae_k1000.py` | `scripts/archive/rwy14-review-v1/source/replace_tcvae_k1000.py` |
| `runs/rwy14-review-v1/source/test_trajectory_rules.py` | `scripts/archive/rwy14-review-v1/source/test_trajectory_rules.py` |
| `runs/rwy14-review-v1/source/trajectory_rules.html` | `scripts/archive/rwy14-review-v1/source/trajectory_rules.html` |
| `runs/rwy14-review-v1/source/trajectory_rules.py` | `scripts/archive/rwy14-review-v1/source/trajectory_rules.py` |
| `runs/rwy14-review-v1/source/verify_review.py` | `scripts/verify_review.py` |
| `runs/rwy14-review-v1/source/verify_training_compare.py` | `scripts/verify_training_compare.py` |
| `runs/tianjin-bad128-gap-audit-20261008/audit_gaps.py` | `scripts/audit_gaps.py` |
| `runs/tianjin-preprocess-20261003/summarize_runways.py` | `scripts/summarize_runways.py` |
| `runs/tianjin-rwy16l-review-20261004/review_tianjin_tcvae.py` | `scripts/archive/tianjin-rwy16l-review-20261004/review_tianjin_tcvae.py` |
| `runs/tianjin-rwy16l-review-20261004-v2/review_tianjin_tcvae.py` | `scripts/archive/tianjin-rwy16l-review-20261004-v2/review_tianjin_tcvae.py` |
| `runs/tianjin-track-unwrap-audit-20261007/audit_tianjin_track.py` | `scripts/audit_tianjin_track.py` (identical copy retained) |
| `runs/tianjin-track-unwrap-audit-20261007/compress_transfer.py` | `scripts/archive/tianjin-track-unwrap-audit-20261007/compress_transfer.py` |
| `runs/tianjin-track-unwrap-audit-20261007/inspect_environment.py` | `scripts/inspect_environment.py` |
| `runs/tianjin-track-unwrap-audit-20261007/plot_tianjin_gps_integration.py` | `scripts/plot_tianjin_gps_integration.py` (identical copy retained) |
| `runs/tianjin-track-unwrap-audit-20261007/repair_tianjin_track.py` | `scripts/repair_tianjin_track.py` (identical copy retained) |
| `runs/zurich-rwy14-ddpm-resume700-20261008/audit_epoch500.py` | `scripts/audit_epoch500.py` |
| `runs/zurich_rwy14_ddpm_unconditional-v1-seed-42/diffusion-process-n100-steps1000-seed42-20261009/refined-late-stages/refine_ddpm_denoising.py` | `scripts/archive/plot-lonlat-20261010/source-before/scripts/refine_ddpm_denoising.py` (identical copy retained) |
| `runs/zurich_rwy14_flow162_216_latent_ddpm-v1-seed-42/pca_posterior_inputs_seed42/export_script_used.py` | `scripts/archive/zurich_rwy14_flow162_216_latent_ddpm-v1-seed-42/pca_posterior_inputs_seed42/export_script_used.py` |
| `runs/zurich_rwy14_flow162_216_tcvae_vamp_k256-v1-seed-42/diagnostics/pca-matched-trajectory-comparison-n100-seed42/source_script.py` | `scripts/compare_vamp_trajectories.py` (identical copy retained) |
| `runs/zurich_rwy14_flow162_216_tcvae_vamp_k256-v1-seed-42/latent_analysis/pca-inputs-n500-seed42/export_script_used.py` | `scripts/archive/zurich_rwy14_flow162_216_tcvae_vamp_k256-v1-seed-42/latent_analysis/pca-inputs-n500-seed42/export_script_used.py` |
| `runs/zurich_rwy14_flow162_216_tcvae_vamp_k256-v1-seed-42/latent_analysis/pca-inputs-n500-seed42/tcvae_source.py` | `scripts/archive/zurich_rwy14_flow162_216_tcvae_vamp_k256-v1-seed-42/latent_analysis/pca-inputs-n500-seed42/tcvae_source.py` |
| `runs/zurich_rwy14_flow162_216_tcvae_vamp_k4-v1-seed-42/component_diagnostics.py` | `scripts/component_diagnostics.py` |

The calculation source at `archive/distribution-evaluation-20261008/evaluate_generation_distributions.py` is deliberately retained once: existing distribution manifests identify this exact source hash. Replotting now loads recorded calculation versions from `scripts/`, and new outputs do not embed executable script copies.
