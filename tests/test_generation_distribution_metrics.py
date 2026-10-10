"""Numerical contracts for the standalone fixed-input distribution evaluation."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np

SOURCE = (
    Path(__file__).resolve().parents[1] / "scripts/evaluate_generation_distributions.py"
)
SPEC = importlib.util.spec_from_file_location("distribution_evaluation", SOURCE)
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)

REPLOT_SPEC = importlib.util.spec_from_file_location(
    "distribution_replot", SOURCE.with_name("replot_generation_distributions.py")
)
REPLOT = importlib.util.module_from_spec(REPLOT_SPEC)
REPLOT_SPEC.loader.exec_module(REPLOT)


def test_replot_uses_recorded_source_after_relocation(tmp_path):
    current = tmp_path / "scripts/evaluate_generation_distributions.py"
    archived = tmp_path / "scripts/archive/old/evaluate_generation_distributions.py"
    current.parent.mkdir(parents=True)
    archived.parent.mkdir(parents=True)
    current.write_text("new calculation", encoding="utf8")
    archived.write_text("recorded calculation", encoding="utf8")
    expected = hashlib.sha256(archived.read_bytes()).hexdigest()
    manifest = {"input_sha256": {current.relative_to(tmp_path).as_posix(): expected}}
    assert REPLOT.evaluation_source(tmp_path, manifest) == archived
    current_sha = hashlib.sha256(current.read_bytes()).hexdigest()
    manifest["input_sha256"] = {current.relative_to(tmp_path).as_posix(): current_sha}
    assert REPLOT.evaluation_source(tmp_path, manifest) == current


def test_track_changes_use_shortest_circular_difference():
    actual = METRICS.circular_delta(np.array([[359, 1, 359], [1, 359, 1]]))
    np.testing.assert_array_equal(actual, [[2, -2], [-2, 2]])


def test_wasserstein_retains_units_and_reports_missing_values():
    result = METRICS.compare_1d(
        np.array([1, 2, 3]), np.array([4, 5, 6, np.nan]), np.random.default_rng(42)
    )
    np.testing.assert_allclose(result["wasserstein"], 3)
    assert result["n_generated"] == 3
    assert result["ks_statistic"] == 1


def test_invalid_raw_time_excludes_rate_but_preserves_shape_features():
    values = np.zeros((2, 4, 4))
    values[:, :, 0] = [359, 1, 3, 5]
    values[:, :, 1] = [300, 290, 280, 270]
    values[:, :, 2] = [1000, 900, 800, 700]
    values[0, :, 3] = [0, 1, 2, 3]
    values[1, :, 3] = [0, 1, 1, 2]
    result = METRICS.trajectory_features(values)
    np.testing.assert_array_equal(result["total_abs_turn"], [6, 6])
    assert result["p95_abs_vertical_rate"][0] == 6000
    assert np.isnan(result["p95_abs_vertical_rate"][1])
    assert np.isnan(result["time_step_cv"][1])
    assert np.isfinite(result["p95_abs_altitude_step"]).all()


def test_nearest_real_baseline_excludes_self():
    real = np.array([[0.0], [2.0], [4.0]])
    result, to_generated, to_real, baseline = METRICS.distance_diagnostics(real, real)
    np.testing.assert_array_equal(baseline, [2, 2, 2])
    np.testing.assert_array_equal(to_generated, [0, 0, 0])
    np.testing.assert_array_equal(to_real, [0, 0, 0])
    assert result["diversity_pair_distance_ratio"] == 1


def test_envelope_threshold_is_calibrated_on_validation():
    train = np.zeros((10, 4, 4))
    train[:, :, 1:4] = np.arange(10)[:, None, None]
    validation = np.full((20, 4, 4), 4.0)
    validation[-1, :2, 1:4] = 20
    fitted = METRICS.calibrated_envelope(train, validation)
    np.testing.assert_allclose(fitted["lower"], 0.09)
    np.testing.assert_allclose(fitted["upper"], 8.91)
    assert fitted["threshold"] == 0.5
    score, outside = METRICS.envelope_scores(
        validation, fitted["lower"], fitted["upper"]
    )
    assert outside.shape == (20, 4, 3)
    assert score[-1] == 0.5


def test_time_origin_and_angle_branch_do_not_change_joint_distances():
    values = np.ones((2, 4, 4))
    values[0, :, 0] = [359, 361, 363, 365]
    values[1, :, 0] = [-1, 1, 3, 5]
    values[0, :, 3] = [100, 101, 102, 103]
    values[1, :, 3] = [200, 201, 202, 203]
    joint = METRICS.joint_representation(
        METRICS.relative_time(values), fixed_scale=True
    )
    np.testing.assert_allclose(joint[0], joint[1], atol=1e-14)
