import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "trajectory_rules", Path(__file__).parents[1] / "scripts/trajectory_rules.py"
)
rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rules)


def test_circular_boundary_and_translation():
    assert abs(rules.angle(rules.center([359, 1]), 0)) < 1e-10
    xy = np.array([[10.0, 30.0], [15.0, 18.0], [20.0, 10.0]])
    np.testing.assert_allclose(rules.remaining(xy), rules.remaining(xy + 10000))


def test_score_saturates_and_failure_cannot_pass():
    upper = {"kind": "upper", "good": 2.0, "fail": 6.0}
    assert rules.item_score(1, upper) == (True, 100.0)
    assert rules.item_score(4, upper) == (True, 50.0)
    assert rules.item_score(6, upper) == (True, 0.0)
    assert rules.item_score(6.01, upper) == (False, 0.0)
    lower = {"kind": "lower", "good": 100.0, "fail": 50.0}
    assert rules.item_score(1000, lower) == (True, 100.0)
    assert rules.item_score(49, lower) == (False, 0.0)


def test_suffix_cannot_use_remote_alignment_and_limits_noise():
    segments = np.ones(4) * 100
    dt = np.ones(4)
    assert rules.suffix_length(np.array([0, 0, 0, 20, 20]), segments, dt, 3, 5, 1) == 0
    assert rules.suffix_length(np.array([0, 0, 4, 0, 0]), segments, dt, 3, 5, 2) == 400
    assert rules.suffix_length(np.array([0, 0, 4, 0, 0]), segments, dt, 3, 5, 1) == 200


def test_duration_uses_actual_time_and_reversals_need_repetition():
    assert rules.durations(
        np.array([True, True, False, True]), np.array([1.0, 5.0, 2.0, 3.0])
    ) == (9.0, 6.0)
    assert (
        rules.speed_reversals(
            np.array([100.0, 110.0, 100.0, 90.0]), np.arange(4.0) * 10, 5, 100
        )
        == 0
    )
    assert (
        rules.speed_reversals(
            np.array([100.0, 110.0, 100.0, 110.0, 100.0]), np.arange(5.0) * 10, 5, 100
        )
        > 0
    )


def test_equal_bound_has_no_division_by_zero():
    assert rules.item_score(0, {"kind": "upper", "good": 0.0, "fail": 0.0}) == (
        True,
        100.0,
    )
    assert rules.item_score(1, {"kind": "upper", "good": 0.0, "fail": 0.0}) == (
        False,
        0.0,
    )


def test_necessary_gate_failure_cannot_be_offset_by_score():
    names = [name for group in rules.GROUPS.values() for name in group]
    measures = dict.fromkeys(names, 0.0)
    bounds = {name: {"kind": "upper", "good": 1.0, "fail": 2.0} for name in names}
    measures["path_length_m"] = 0.0
    bounds["path_length_m"] = {"kind": "lower", "good": 100.0, "fail": 50.0}
    _, _, failed, total = rules.evaluate(measures, {"bounds": bounds})
    assert total == 100.0
    assert failed == ["path_length_m"]


def test_json_numeric_tolerance_and_missing_far_coverage():
    import json

    json.dumps({"tolerance": rules.numerical_tolerance(np.array([1000.0]))})
    profile = rules.profile(
        np.array([20.0, 10.0, 0.0]),
        np.array([3.0, 2.0, 1.0]),
        np.array([0.0, 10.0, 30.0]),
    )
    np.testing.assert_equal(profile, np.array([1.0, 2.0, np.nan]))
    assert abs(rules.angle(rules.center([359.0, np.nan, 1.0]), 0)) < 1e-10
