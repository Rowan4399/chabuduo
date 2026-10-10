from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import pytest
from cartes.crs import Mercator, PlateCarree
from matplotlib.figure import Figure
from traffic.core import Traffic

from chabuduo import plotting
from chabuduo.artifacts import Normalization, SplitData


@pytest.fixture
def geography(monkeypatch):
    threshold = SimpleNamespace(name="14", longitude=8.55, latitude=47.47)
    airport = SimpleNamespace(
        longitude=8.54, latitude=47.46,
        runways={"14": SimpleNamespace(tuple_runway=[threshold])},
    )
    monkeypatch.setattr(plotting, "airports", {"LSZH": airport})
    projection = pyproj.Proj(
        proj="aeqd", lon_0=airport.longitude, lat_0=airport.latitude,
        datum="WGS84", units="m",
    )
    monkeypatch.setattr(plotting, "airport_projection", lambda _: projection)
    return airport, threshold, projection


def test_geographic_columns_take_precedence_over_xy(geography):
    frame = pd.DataFrame({
        "longitude": [8.5, 8.6], "latitude": [47.4, 47.5],
        "x": [999999, 999999], "y": [999999, 999999],
    })
    assert plotting._geographic_frame(frame, "LSZH") is frame


def test_xy_inverse_preserves_positions_and_metres(geography):
    _, _, projection = geography
    longitude, latitude = np.array([8.3, 8.7]), np.array([47.2, 47.7])
    x, y = projection(longitude, latitude)
    frame = pd.DataFrame({"x": x, "y": y})
    result = plotting._geographic_frame(frame, "LSZH")
    np.testing.assert_allclose(result.longitude, longitude, atol=1e-8)
    np.testing.assert_allclose(result.latitude, latitude, atol=1e-8)
    pd.testing.assert_frame_equal(result[["x", "y"]], frame)
    with pytest.raises(ValueError, match="longitude/latitude"):
        plotting._geographic_frame(pd.DataFrame({"x": x}), "LSZH")


def test_cached_threshold_offsets_are_inverted_without_resampling(
    geography, monkeypatch
):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import plot_ddpm_diffusion_stages as stages

    airport, threshold, projection = geography
    monkeypatch.setattr(stages, "airports", {"LSZH": airport})
    monkeypatch.setattr(stages, "airport_projection", lambda _: projection)
    longitude = np.array([8.4, 8.5, threshold.longitude])
    latitude = np.array([47.3, 47.4, threshold.latitude])
    x, y = projection(longitude, latitude)
    anchor = np.array(projection(threshold.longitude, threshold.latitude))
    offsets = ((np.column_stack((x, y)) - anchor) / 1000).reshape(1, 1, 3, 2)
    original = offsets.copy()
    converted = stages.map_lonlat(offsets)
    np.testing.assert_allclose(converted[0, 0], np.column_stack((longitude, latitude)))
    np.testing.assert_array_equal(offsets, original)
    assert converted.shape == offsets.shape


def test_plot_reads_lonlat_only_parquet_and_shares_extent(
    tmp_path, geography, monkeypatch
):
    config = SimpleNamespace(
        name="synthetic", preprocess=SimpleNamespace(airport="LSZH", runways=["14"]),
        prepare=SimpleNamespace(representation=SimpleNamespace(anchor="runway_threshold")),
    )
    frame = pd.DataFrame({
        "flight_id": ["a", "a", "b", "b"],
        "timestamp": pd.date_range("2026-01-01", periods=4, freq="s", tz="UTC"),
        "longitude": [8.3, 8.55, 8.7, 8.55],
        "latitude": [47.2, 47.47, 47.7, 47.47],
    })
    frame.to_parquet(tmp_path / "generated.parquet", index=False)
    values = np.zeros((2, 2, 1), dtype=np.float32)
    SplitData(values, np.array([[0, 1], [0, 1]], dtype=np.float32),
              np.array(["a", "b"]), np.array(["a", "b"])).save(tmp_path / "test.npz")
    Normalization(np.zeros((2, 1)), np.ones((2, 1))).save(
        tmp_path / "normalization.json"
    )
    monkeypatch.setattr(plotting, "build_generated_traffic", lambda *_: Traffic(frame))
    captured = []
    savefig = Figure.savefig

    def inspect_map(figure, *args, **kwargs):
        captured.extend(figure.axes)
        savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect_map)
    output = plotting.plot_samples(tmp_path, tmp_path, config, count=2)
    assert output.is_file() and output.stat().st_size > 1000
    for axis in captured:
        assert isinstance(axis.projection, Mercator)
        np.testing.assert_array_equal(axis.lines[0].get_xdata(), [8.3, 8.55])
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), [47.2, 47.47])
        np.testing.assert_allclose(axis.collections[0].get_offsets(), [[8.55, 47.47]])
        assert axis.get_xlabel() == "Longitude (°)"
    np.testing.assert_allclose(captured[0].get_extent(PlateCarree()),
                               captured[1].get_extent(PlateCarree()))
    assert not plt.get_fignums()
