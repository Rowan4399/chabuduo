# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from traffic.data import airports

from chabuduo.artifacts import Normalization, SplitData
from chabuduo.prepare import airport_projection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "zurich_rwy14_flow162_216_ddpm_unconditional-v1-seed-42"
SAMPLE_NAME = "resume700-best-seed-42-n-100-steps-100-eta-0"
ARTIFACT_NAME = "zurich_rwy14_flow162_216_ddpm_unconditional-v1"
KNOT_TO_METRES_PER_SECOND = 0.514444
METRES_PER_NAUTICAL_MILE = 1852.0


def remaining_distance_nm(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    segment_m = np.hypot(np.diff(x_m), np.diff(y_m))
    remaining_m = np.concatenate(
        (np.cumsum(segment_m[::-1])[::-1], np.zeros(1, dtype=float))
    )
    return remaining_m / METRES_PER_NAUTICAL_MILE


def kinematic_remaining_distance_nm(
    speed_kt: np.ndarray, elapsed_time: np.ndarray
) -> np.ndarray:
    delta_time = np.maximum(np.diff(elapsed_time), 0.0)
    mean_speed = 0.5 * (speed_kt[:-1] + speed_kt[1:])
    segment_nm = mean_speed * delta_time / 3600.0
    return np.concatenate(
        (np.cumsum(segment_nm[::-1])[::-1], np.zeros(1, dtype=float))
    )


def interpolate_profile(
    remaining_nm: np.ndarray,
    values: np.ndarray,
    grid_nm: np.ndarray,
) -> np.ndarray:
    distance = remaining_nm[::-1]
    profile = values[::-1]
    unique_distance, unique_indices = np.unique(distance, return_index=True)
    unique_profile = profile[unique_indices]
    if len(unique_distance) < 2:
        return np.full_like(grid_nm, np.nan, dtype=float)
    return np.interp(
        grid_nm,
        unique_distance,
        unique_profile,
        left=np.nan,
        right=np.nan,
    )


def round_list(values: np.ndarray, digits: int = 3) -> list[float | None]:
    rounded = np.round(values.astype(float), digits)
    return [None if not np.isfinite(value) else float(value) for value in rounded]


def runway_geometry() -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    airport = airports["LSZH"]
    if airport is None:
        raise ValueError("Unknown airport LSZH")
    runway = airport.runways["14"]
    threshold = next(point for point in runway.tuple_runway if point.name == "14")
    projection = airport_projection("LSZH")
    longitudes = [point.longitude for point in runway.tuple_runway]
    latitudes = [point.latitude for point in runway.tuple_runway]
    runway_x, runway_y = projection(longitudes, latitudes)
    threshold_x, threshold_y = projection(threshold.longitude, threshold.latitude)
    return (
        np.asarray(runway_x, dtype=float),
        np.asarray(runway_y, dtype=float),
        (float(threshold_x), float(threshold_y)),
    )


def reconstruct_reference_path(
    track_deg: np.ndarray,
    speed_kt: np.ndarray,
    elapsed_time: np.ndarray,
    threshold_xy: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    delta_time = np.maximum(np.diff(elapsed_time), 0.0)
    mean_speed = 0.5 * (speed_kt[:-1] + speed_kt[1:])
    segment_m = mean_speed * KNOT_TO_METRES_PER_SECOND * delta_time
    bearing = np.radians(track_deg[1:] % 360.0)
    delta_x = segment_m * np.sin(bearing)
    delta_y = segment_m * np.cos(bearing)
    x_m = np.empty_like(track_deg, dtype=float)
    y_m = np.empty_like(track_deg, dtype=float)
    x_m[-1], y_m[-1] = threshold_xy
    for index in range(len(track_deg) - 2, -1, -1):
        x_m[index] = x_m[index + 1] - delta_x[index]
        y_m[index] = y_m[index + 1] - delta_y[index]
    return x_m, y_m


def generated_trajectories(path: Path) -> list[dict[str, object]]:
    frame = pd.read_parquet(path)
    sort_column = "timestamp" if "timestamp" in frame else "timedelta"
    trajectories: list[dict[str, object]] = []
    for trajectory_id, trajectory in frame.groupby("flight_id", sort=False):
        trajectory = trajectory.sort_values(sort_column)
        x_m = trajectory["x"].to_numpy(dtype=float)
        y_m = trajectory["y"].to_numpy(dtype=float)
        altitude_ft = trajectory["altitude"].to_numpy(dtype=float)
        speed_kt = trajectory["groundspeed"].to_numpy(dtype=float)
        remaining_nm = remaining_distance_nm(x_m, y_m)
        trajectories.append(
            {
                "id": str(trajectory_id),
                "xKm": round_list(x_m / 1000.0),
                "yKm": round_list(y_m / 1000.0),
                "remainingNm": round_list(remaining_nm),
                "altitudeFt": round_list(altitude_ft, 1),
                "speedKt": round_list(speed_kt, 1),
                "durationSec": round(
                    float(trajectory["elapsed_time"].iloc[-1]), 1
                ),
                "pathLengthNm": round(float(remaining_nm[0]), 2),
            }
        )
    return trajectories


def reference_data(
    artifact: Path,
    threshold_xy: tuple[float, float],
    generated_max_nm: float,
) -> tuple[dict[str, object], list[dict[str, list[float | None]]]]:
    split = SplitData.load(artifact / "train.npz")
    normalization = Normalization.load(artifact / "normalization.json")
    values = normalization.inverse_transform(split.values)
    speed = values[:, :, 1].astype(float)
    altitude = values[:, :, 2].astype(float)
    track = values[:, :, 0].astype(float)
    elapsed = split.elapsed_time.astype(float)
    remaining = np.asarray(
        [
            kinematic_remaining_distance_nm(speed[index], elapsed[index])
            for index in range(len(values))
        ]
    )
    maximum_nm = max(generated_max_nm, float(np.percentile(remaining[:, 0], 99)))
    maximum_nm = min(70.0, max(40.0, np.ceil(maximum_nm / 5.0) * 5.0))
    grid_nm = np.linspace(0.0, maximum_nm, 161)
    altitude_profiles = np.asarray(
        [
            interpolate_profile(remaining[index], altitude[index], grid_nm)
            for index in range(len(values))
        ]
    )
    speed_profiles = np.asarray(
        [
            interpolate_profile(remaining[index], speed[index], grid_nm)
            for index in range(len(values))
        ]
    )

    def envelope(profiles: np.ndarray) -> dict[str, list[float | None]]:
        return {
            "low": round_list(np.nanpercentile(profiles, 5, axis=0), 1),
            "median": round_list(np.nanpercentile(profiles, 50, axis=0), 1),
            "high": round_list(np.nanpercentile(profiles, 95, axis=0), 1),
        }

    reference_indices = np.linspace(
        0, len(values) - 1, min(12, len(values)), dtype=int
    )
    map_references: list[dict[str, list[float | None]]] = []
    for index in reference_indices:
        x_m, y_m = reconstruct_reference_path(
            track[index], speed[index], elapsed[index], threshold_xy
        )
        map_references.append(
            {
                "xKm": round_list(x_m / 1000.0),
                "yKm": round_list(y_m / 1000.0),
            }
        )
    return (
        {
            "gridNm": round_list(grid_nm),
            "altitude": envelope(altitude_profiles),
            "speed": envelope(speed_profiles),
            "count": int(len(values)),
        },
        map_references,
    )


def padded_extent(
    trajectories: list[dict[str, object]],
    reference_paths: list[dict[str, list[float | None]]],
    runway_x_km: np.ndarray,
    runway_y_km: np.ndarray,
) -> list[float]:
    x_values = [
        value
        for trajectory in trajectories
        for value in trajectory["xKm"]  # type: ignore[index]
        if value is not None
    ]
    y_values = [
        value
        for trajectory in trajectories
        for value in trajectory["yKm"]  # type: ignore[index]
        if value is not None
    ]
    for path in reference_paths:
        x_values.extend(value for value in path["xKm"] if value is not None)
        y_values.extend(value for value in path["yKm"] if value is not None)
    x_values.extend(runway_x_km.tolist())
    y_values.extend(runway_y_km.tolist())
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    padding = 0.05 * max(x_max - x_min, y_max - y_min)
    return [
        round(x_min - padding, 2),
        round(x_max + padding, 2),
        round(y_min - padding, 2),
        round(y_max + padding, 2),
    ]


def build_payload(generated: Path, artifact: Path) -> dict[str, object]:
    trajectories = generated_trajectories(generated)
    runway_x, runway_y, threshold_xy = runway_geometry()
    generated_max_nm = max(
        float(trajectory["pathLengthNm"]) for trajectory in trajectories
    )
    profiles, references = reference_data(
        artifact, threshold_xy, generated_max_nm
    )
    runway_x_km = runway_x / 1000.0
    runway_y_km = runway_y / 1000.0
    return {
        "runName": RUN_NAME,
        "sampleName": SAMPLE_NAME,
        "projection": "Airport-centred azimuthal equidistant (AEQD)",
        "trajectories": trajectories,
        "referenceProfiles": profiles,
        "referencePaths": references,
        "mapExtent": padded_extent(
            trajectories, references, runway_x_km, runway_y_km
        ),
        "runway": {
            "xKm": round_list(runway_x_km),
            "yKm": round_list(runway_y_km),
            "thresholdXKm": round(threshold_xy[0] / 1000.0, 3),
            "thresholdYKm": round(threshold_xy[1] / 1000.0, 3),
        },
    }


HTML_TEMPLATE = r"""
<div id="trajectory-reviewer">
  <style>
    #trajectory-reviewer {
      color: light-dark(#172033, #e7edf8);
      background: light-dark(#f6f8fc, #101521);
      font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
      min-height: 100vh;
      padding: 18px;
      box-sizing: border-box;
    }
    #trajectory-reviewer * { box-sizing: border-box; }
    #trajectory-reviewer header,
    #trajectory-reviewer .toolbar,
    #trajectory-reviewer .summary,
    #trajectory-reviewer .decision-row,
    #trajectory-reviewer .navigation {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    #trajectory-reviewer header { justify-content: space-between; margin-bottom: 12px; }
    #trajectory-reviewer h1,
    #trajectory-reviewer h2,
    #trajectory-reviewer h3 { margin: 0; font-weight: 500; }
    #trajectory-reviewer h1 { font-size: 20px; }
    #trajectory-reviewer h2 { font-size: 17px; }
    #trajectory-reviewer h3 { font-size: 14px; }
    #trajectory-reviewer button,
    #trajectory-reviewer select,
    #trajectory-reviewer textarea,
    #trajectory-reviewer input {
      font: inherit;
    }
    #trajectory-reviewer button,
    #trajectory-reviewer select {
      color: inherit;
      background: light-dark(#ffffff, #1a2232);
      border: 1px solid light-dark(#cdd5e3, #344157);
      border-radius: 6px;
      padding: 7px 10px;
    }
    #trajectory-reviewer button { cursor: pointer; }
    #trajectory-reviewer button.active {
      color: light-dark(#ffffff, #0c111b);
      background: light-dark(#273a5f, #dbe7ff);
    }
    #trajectory-reviewer button.keep.active { background: #2f855a; color: #fff; }
    #trajectory-reviewer button.review.active { background: #b7791f; color: #fff; }
    #trajectory-reviewer button.reject.active { background: #c53030; color: #fff; }
    #trajectory-reviewer .summary span {
      padding: 4px 8px;
      border-radius: 999px;
      background: light-dark(#e8edf6, #222c3e);
    }
    #trajectory-reviewer .toolbar { margin: 10px 0 14px; }
    #trajectory-reviewer .spacer { flex: 1; }
    #trajectory-reviewer .overview-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(150px, 1fr));
      gap: 10px;
    }
    #trajectory-reviewer .tile {
      color: inherit;
      text-align: left;
      padding: 7px;
      background: light-dark(#ffffff, #171e2c);
      border: 2px solid light-dark(#dbe1eb, #2d394e);
      border-radius: 7px;
    }
    #trajectory-reviewer .tile[data-status="keep"] { border-color: #2f855a; }
    #trajectory-reviewer .tile[data-status="review"] { border-color: #b7791f; }
    #trajectory-reviewer .tile[data-status="reject"] { border-color: #c53030; }
    #trajectory-reviewer .tile-title {
      display: flex;
      justify-content: space-between;
      margin-bottom: 4px;
      font-size: 12px;
    }
    #trajectory-reviewer .tile svg { width: 100%; height: 120px; display: block; }
    #trajectory-reviewer .detail-layout {
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(300px, 1fr);
      gap: 14px;
      align-items: start;
    }
    #trajectory-reviewer .plot {
      background: light-dark(#ffffff, #171e2c);
      border: 1px solid light-dark(#dbe1eb, #2d394e);
      border-radius: 7px;
      padding: 8px;
    }
    #trajectory-reviewer .plot + .plot { margin-top: 12px; }
    #trajectory-reviewer .plot svg { width: 100%; display: block; }
    #trajectory-reviewer #map-svg { height: 620px; }
    #trajectory-reviewer .profile-svg { height: 270px; }
    #trajectory-reviewer .assessment {
      margin-top: 14px;
      padding: 12px;
      background: light-dark(#ffffff, #171e2c);
      border: 1px solid light-dark(#dbe1eb, #2d394e);
      border-radius: 7px;
    }
    #trajectory-reviewer .ratings {
      display: grid;
      grid-template-columns: repeat(3, minmax(150px, 1fr));
      gap: 10px;
      margin: 12px 0;
    }
    #trajectory-reviewer label { display: grid; gap: 5px; font-size: 13px; }
    #trajectory-reviewer .reasons {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      margin: 12px 0;
    }
    #trajectory-reviewer .reasons label { display: flex; align-items: center; gap: 5px; }
    #trajectory-reviewer textarea {
      width: 100%;
      min-height: 72px;
      resize: vertical;
      color: inherit;
      background: light-dark(#ffffff, #111725);
      border: 1px solid light-dark(#cdd5e3, #344157);
      border-radius: 6px;
      padding: 8px;
    }
    #trajectory-reviewer .muted { color: light-dark(#637089, #aeb9cc); font-size: 12px; }
    #trajectory-reviewer .hidden { display: none !important; }
    #trajectory-reviewer svg text { fill: currentColor; font-size: 11px; }
    #trajectory-reviewer .frame { fill: none; stroke: light-dark(#bbc5d4, #435069); }
    #trajectory-reviewer .gridline { stroke: light-dark(#e4e8ef, #2a3447); stroke-width: 1; }
    #trajectory-reviewer .generated { fill: none; stroke: #e45756; stroke-width: 2.2; }
    #trajectory-reviewer .reference { fill: none; stroke: light-dark(#7d8798, #8995aa); stroke-width: 1; opacity: 0.25; }
    #trajectory-reviewer .runway { stroke: light-dark(#202938, #f1f5fb); stroke-width: 3; }
    #trajectory-reviewer .median { fill: none; stroke: #4c78a8; stroke-width: 2; }
    #trajectory-reviewer .band { fill: #4c78a8; opacity: 0.16; }
    #trajectory-reviewer .threshold { fill: #2f855a; }
    @media (max-width: 900px) {
      #trajectory-reviewer .overview-grid { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
      #trajectory-reviewer .detail-layout { grid-template-columns: 1fr; }
      #trajectory-reviewer #map-svg { height: 470px; }
    }
  </style>

  <header>
    <div>
      <h1>RWY14 / flow 162–216 人工轨迹审核</h1>
      <div class="muted" id="subtitle"></div>
    </div>
    <div class="summary" id="summary"></div>
  </header>

  <div class="toolbar">
    <button type="button" id="show-overview" class="active">第一轮：接触表</button>
    <button type="button" id="show-detail">第二轮：逐轨迹</button>
    <span class="spacer"></span>
    <button type="button" id="export-csv">导出审核 CSV</button>
  </div>

  <section id="overview-panel">
    <div class="toolbar">
      <button type="button" id="previous-page">上一页</button>
      <span id="page-label"></span>
      <button type="button" id="next-page">下一页</button>
      <span class="muted">点击小图进入逐轨迹审核；边框显示已保存决定。</span>
    </div>
    <div class="overview-grid" id="overview-grid"></div>
  </section>

  <section id="detail-panel" class="hidden">
    <div class="navigation">
      <button type="button" id="previous-trajectory">上一条</button>
      <select id="trajectory-select" aria-label="轨迹"></select>
      <button type="button" id="next-trajectory">下一条</button>
      <label style="display:flex;align-items:center;gap:6px">
        <input type="checkbox" id="show-references">
        显示12条固定真实参考轨迹
      </label>
      <span class="muted" id="trajectory-meta"></span>
    </div>
    <div class="detail-layout">
      <div class="plot">
        <h2>水平轨迹</h2>
        <div class="muted">机场中心AEQD投影；横纵坐标单位km，所有轨迹固定范围。</div>
        <svg id="map-svg" role="img" aria-label="当前生成轨迹地图"></svg>
      </div>
      <div>
        <div class="plot">
          <h2>高度剖面</h2>
          <div class="muted">高度ft MSL；阴影为真实训练集5%–95%，蓝线为中位数。</div>
          <svg id="altitude-svg" class="profile-svg" role="img" aria-label="高度剖面"></svg>
        </div>
        <div class="plot">
          <h2>速度剖面</h2>
          <div class="muted">地速kt；横轴为沿实际路径到跑道入口的剩余距离。</div>
          <svg id="speed-svg" class="profile-svg" role="img" aria-label="速度剖面"></svg>
        </div>
      </div>
    </div>

    <div class="assessment">
      <h2 id="assessment-title"></h2>
      <div class="ratings">
        <label>水平形态
          <select id="horizontal-rating">
            <option value="">未评</option><option value="2">合理</option>
            <option value="1">边缘</option><option value="0">异常</option>
          </select>
        </label>
        <label>高度剖面
          <select id="altitude-rating">
            <option value="">未评</option><option value="2">合理</option>
            <option value="1">边缘</option><option value="0">异常</option>
          </select>
        </label>
        <label>速度剖面
          <select id="speed-rating">
            <option value="">未评</option><option value="2">合理</option>
            <option value="1">边缘</option><option value="0">异常</option>
          </select>
        </label>
      </div>
      <div class="decision-row">
        <strong>总体决定</strong>
        <button type="button" class="keep" data-decision="keep">Keep</button>
        <button type="button" class="review" data-decision="review">Review</button>
        <button type="button" class="reject" data-decision="reject">Reject</button>
        <span class="muted">快捷键：K / R / X</span>
      </div>
      <div class="reasons" id="reasons"></div>
      <label>备注
        <textarea id="notes" placeholder="记录可疑位置或保留/剔除理由"></textarea>
      </label>
    </div>
  </section>

  <script>
    (() => {
      const DATA = __DATA__;
      const root = document.getElementById("trajectory-reviewer");
      const storageKey = `trajectory-review:${DATA.runName}:${DATA.sampleName}:manual-v1`;
      const reasonOptions = [
        ["horizontal_zigzag", "水平锯齿"],
        ["unexpected_reversal", "异常折返"],
        ["unusual_vectoring", "异常绕行"],
        ["poor_runway_alignment", "跑道对准异常"],
        ["altitude_profile", "高度剖面"],
        ["speed_profile", "速度剖面"],
        ["near_duplicate", "近似重复"],
        ["duration_outlier", "持续时间"],
        ["uncertain", "无法确定"]
      ];
      let state = {};
      try { state = JSON.parse(localStorage.getItem(storageKey) || "{}"); } catch (_) {}
      let currentIndex = 0;
      let currentPage = 0;
      const pageSize = 20;

      const $ = (selector) => root.querySelector(selector);
      const svgNS = "http://www.w3.org/2000/svg";
      const makeSvg = (name, attrs = {}) => {
        const node = document.createElementNS(svgNS, name);
        Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
        return node;
      };
      const pathData = (xs, ys, xScale, yScale) => xs.map((x, index) =>
        `${index ? "L" : "M"}${xScale(x).toFixed(2)},${yScale(ys[index]).toFixed(2)}`
      ).join(" ");
      const save = () => localStorage.setItem(storageKey, JSON.stringify(state));
      const recordFor = (id) => state[id] || {
        horizontal: "", altitude: "", speed: "", decision: "", reasons: [], notes: ""
      };

      function updateSummary() {
        const counts = { keep: 0, review: 0, reject: 0, unreviewed: 0 };
        DATA.trajectories.forEach((trajectory) => {
          const decision = recordFor(trajectory.id).decision;
          if (decision) counts[decision] += 1; else counts.unreviewed += 1;
        });
        $("#summary").innerHTML = `
          <span>Keep ${counts.keep}</span><span>Review ${counts.review}</span>
          <span>Reject ${counts.reject}</span><span>未评 ${counts.unreviewed}</span>`;
      }

      function scales(extent, width, height, margin) {
        const [xMin, xMax, yMin, yMax] = extent;
        const xScale = (value) => margin + (value - xMin) / (xMax - xMin) * (width - 2 * margin);
        const yScale = (value) => height - margin - (value - yMin) / (yMax - yMin) * (height - 2 * margin);
        return { xScale, yScale };
      }

      function drawMiniMap(svg, trajectory) {
        const width = 190, height = 120, margin = 5;
        svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
        svg.replaceChildren();
        const { xScale, yScale } = scales(DATA.mapExtent, width, height, margin);
        svg.appendChild(makeSvg("path", {
          d: pathData(DATA.runway.xKm, DATA.runway.yKm, xScale, yScale), class: "runway"
        }));
        svg.appendChild(makeSvg("path", {
          d: pathData(trajectory.xKm, trajectory.yKm, xScale, yScale), class: "generated"
        }));
        svg.appendChild(makeSvg("circle", {
          cx: xScale(DATA.runway.thresholdXKm), cy: yScale(DATA.runway.thresholdYKm),
          r: 2.5, class: "threshold"
        }));
      }

      function renderOverview() {
        const grid = $("#overview-grid");
        grid.replaceChildren();
        const start = currentPage * pageSize;
        DATA.trajectories.slice(start, start + pageSize).forEach((trajectory, offset) => {
          const button = document.createElement("button");
          button.type = "button";
          button.className = "tile";
          button.dataset.status = recordFor(trajectory.id).decision;
          const title = document.createElement("div");
          title.className = "tile-title";
          title.innerHTML = `<strong>${trajectory.id}</strong><span>${recordFor(trajectory.id).decision || "未评"}</span>`;
          const svg = makeSvg("svg", { "aria-label": `${trajectory.id} 水平轨迹` });
          button.append(title, svg);
          drawMiniMap(svg, trajectory);
          button.addEventListener("click", () => {
            currentIndex = start + offset;
            showPanel("detail");
            renderDetail();
          });
          grid.appendChild(button);
        });
        const pages = Math.ceil(DATA.trajectories.length / pageSize);
        $("#page-label").textContent = `第 ${currentPage + 1} / ${pages} 页`;
        $("#previous-page").disabled = currentPage === 0;
        $("#next-page").disabled = currentPage === pages - 1;
      }

      function drawAxes(svg, width, height, margin, xTicks, yTicks, xScale, yScale, labels) {
        svg.appendChild(makeSvg("rect", {
          x: margin.left, y: margin.top,
          width: width - margin.left - margin.right,
          height: height - margin.top - margin.bottom,
          class: "frame"
        }));
        xTicks.forEach((tick) => {
          svg.appendChild(makeSvg("line", {
            x1: xScale(tick), x2: xScale(tick), y1: margin.top,
            y2: height - margin.bottom, class: "gridline"
          }));
          const text = makeSvg("text", { x: xScale(tick), y: height - 8, "text-anchor": "middle" });
          text.textContent = tick.toFixed(0); svg.appendChild(text);
        });
        yTicks.forEach((tick) => {
          svg.appendChild(makeSvg("line", {
            x1: margin.left, x2: width - margin.right,
            y1: yScale(tick), y2: yScale(tick), class: "gridline"
          }));
          const text = makeSvg("text", { x: margin.left - 7, y: yScale(tick) + 4, "text-anchor": "end" });
          text.textContent = tick.toFixed(0); svg.appendChild(text);
        });
        const xLabel = makeSvg("text", { x: (margin.left + width - margin.right) / 2, y: height - 1, "text-anchor": "middle" });
        xLabel.textContent = labels.x; svg.appendChild(xLabel);
        const yLabel = makeSvg("text", { x: 12, y: (margin.top + height - margin.bottom) / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${(margin.top + height - margin.bottom) / 2})` });
        yLabel.textContent = labels.y; svg.appendChild(yLabel);
      }

      function renderMap(trajectory) {
        const svg = $("#map-svg");
        const width = 680, height = 620;
        const margin = { left: 55, right: 18, top: 18, bottom: 48 };
        svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
        svg.replaceChildren();
        const [xMin, xMax, yMin, yMax] = DATA.mapExtent;
        const xScale = (value) => margin.left + (value - xMin) / (xMax - xMin) * (width - margin.left - margin.right);
        const yScale = (value) => height - margin.bottom - (value - yMin) / (yMax - yMin) * (height - margin.top - margin.bottom);
        const ticks = (min, max) => Array.from({length: 5}, (_, index) => min + index * (max - min) / 4);
        drawAxes(svg, width, height, margin, ticks(xMin, xMax), ticks(yMin, yMax), xScale, yScale, {x: "Projected x (km)", y: "Projected y (km)"});
        if ($("#show-references").checked) {
          DATA.referencePaths.forEach((path) => svg.appendChild(makeSvg("path", {
            d: pathData(path.xKm, path.yKm, xScale, yScale), class: "reference"
          })));
        }
        svg.appendChild(makeSvg("path", {
          d: pathData(DATA.runway.xKm, DATA.runway.yKm, xScale, yScale), class: "runway"
        }));
        svg.appendChild(makeSvg("path", {
          d: pathData(trajectory.xKm, trajectory.yKm, xScale, yScale), class: "generated"
        }));
        svg.appendChild(makeSvg("circle", {
          cx: xScale(DATA.runway.thresholdXKm), cy: yScale(DATA.runway.thresholdYKm),
          r: 5, class: "threshold"
        }));
      }

      function finiteExtent(values, extras) {
        const all = [...values, ...extras].filter((value) => value !== null && Number.isFinite(value));
        const min = Math.min(...all), max = Math.max(...all);
        const padding = Math.max((max - min) * 0.08, 1);
        return [min - padding, max + padding];
      }

      function renderProfile(svg, trajectory, key, reference, yLabel) {
        const width = 500, height = 270;
        const margin = { left: 58, right: 16, top: 14, bottom: 42 };
        svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
        svg.replaceChildren();
        const maxDistance = Math.max(...DATA.referenceProfiles.gridNm);
        const generatedValues = trajectory[key];
        const yExtent = finiteExtent(generatedValues, [...reference.low, ...reference.high]);
        const xScale = (value) => margin.left + (maxDistance - value) / maxDistance * (width - margin.left - margin.right);
        const yScale = (value) => height - margin.bottom - (value - yExtent[0]) / (yExtent[1] - yExtent[0]) * (height - margin.top - margin.bottom);
        const xTicks = Array.from({length: 6}, (_, index) => maxDistance * index / 5);
        const yTicks = Array.from({length: 5}, (_, index) => yExtent[0] + index * (yExtent[1] - yExtent[0]) / 4);
        drawAxes(svg, width, height, margin, xTicks, yTicks, xScale, yScale, {x: "Remaining path distance (NM)", y: yLabel});
        const grid = DATA.referenceProfiles.gridNm;
        const upper = grid.map((distance, index) => `${index ? "L" : "M"}${xScale(distance).toFixed(2)},${yScale(reference.high[index]).toFixed(2)}`).join(" ");
        const lower = [...grid].reverse().map((distance, reverseIndex) => {
          const index = grid.length - 1 - reverseIndex;
          return `L${xScale(distance).toFixed(2)},${yScale(reference.low[index]).toFixed(2)}`;
        }).join(" ");
        svg.appendChild(makeSvg("path", { d: `${upper} ${lower} Z`, class: "band" }));
        svg.appendChild(makeSvg("path", { d: pathData(grid, reference.median, xScale, yScale), class: "median" }));
        svg.appendChild(makeSvg("path", { d: pathData(trajectory.remainingNm, generatedValues, xScale, yScale), class: "generated" }));
      }

      function loadAssessment(trajectory) {
        const record = recordFor(trajectory.id);
        $("#assessment-title").textContent = `${trajectory.id} 审核记录`;
        $("#horizontal-rating").value = record.horizontal;
        $("#altitude-rating").value = record.altitude;
        $("#speed-rating").value = record.speed;
        $("#notes").value = record.notes;
        root.querySelectorAll("[data-decision]").forEach((button) => {
          button.classList.toggle("active", button.dataset.decision === record.decision);
        });
        root.querySelectorAll("#reasons input").forEach((input) => {
          input.checked = record.reasons.includes(input.value);
        });
      }

      function renderDetail() {
        const trajectory = DATA.trajectories[currentIndex];
        $("#trajectory-select").value = String(currentIndex);
        $("#trajectory-meta").textContent = `${trajectory.id} · ${trajectory.pathLengthNm} NM · ${trajectory.durationSec} s`;
        renderMap(trajectory);
        renderProfile($("#altitude-svg"), trajectory, "altitudeFt", DATA.referenceProfiles.altitude, "Altitude (ft MSL)");
        renderProfile($("#speed-svg"), trajectory, "speedKt", DATA.referenceProfiles.speed, "Groundspeed (kt)");
        loadAssessment(trajectory);
      }

      function updateRecord(mutator) {
        const trajectory = DATA.trajectories[currentIndex];
        const record = {...recordFor(trajectory.id)};
        mutator(record);
        state[trajectory.id] = record;
        save(); updateSummary();
      }

      function showPanel(name) {
        const overview = name === "overview";
        $("#overview-panel").classList.toggle("hidden", !overview);
        $("#detail-panel").classList.toggle("hidden", overview);
        $("#show-overview").classList.toggle("active", overview);
        $("#show-detail").classList.toggle("active", !overview);
        if (overview) renderOverview(); else renderDetail();
      }

      function exportCsv() {
        const header = ["trajectory_id", "horizontal_decision", "altitude_decision", "speed_decision", "overall_decision", "reason_codes", "notes"];
        const escape = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
        const rows = DATA.trajectories.map((trajectory) => {
          const record = recordFor(trajectory.id);
          return [trajectory.id, record.horizontal, record.altitude, record.speed, record.decision, record.reasons.join(";"), record.notes].map(escape).join(",");
        });
        const blob = new Blob(["\ufeff" + [header.join(","), ...rows].join("\r\n")], {type: "text/csv;charset=utf-8"});
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = "manual-review-v1.csv";
        link.click();
        URL.revokeObjectURL(link.href);
      }

      reasonOptions.forEach(([value, label]) => {
        const wrapper = document.createElement("label");
        const input = document.createElement("input");
        input.type = "checkbox"; input.value = value;
        input.addEventListener("change", () => updateRecord((record) => {
          record.reasons = [...root.querySelectorAll("#reasons input:checked")].map((node) => node.value);
        }));
        wrapper.append(input, document.createTextNode(label));
        $("#reasons").appendChild(wrapper);
      });
      DATA.trajectories.forEach((trajectory, index) => {
        const option = document.createElement("option");
        option.value = String(index); option.textContent = trajectory.id;
        $("#trajectory-select").appendChild(option);
      });
      $("#subtitle").textContent = `${DATA.sampleName} · ${DATA.trajectories.length}条 · 真实训练参考${DATA.referenceProfiles.count}条 · ${DATA.projection}`;
      $("#show-overview").addEventListener("click", () => showPanel("overview"));
      $("#show-detail").addEventListener("click", () => showPanel("detail"));
      $("#previous-page").addEventListener("click", () => { currentPage -= 1; renderOverview(); });
      $("#next-page").addEventListener("click", () => { currentPage += 1; renderOverview(); });
      $("#previous-trajectory").addEventListener("click", () => { currentIndex = Math.max(0, currentIndex - 1); renderDetail(); });
      $("#next-trajectory").addEventListener("click", () => { currentIndex = Math.min(DATA.trajectories.length - 1, currentIndex + 1); renderDetail(); });
      $("#trajectory-select").addEventListener("change", (event) => { currentIndex = Number(event.target.value); renderDetail(); });
      $("#show-references").addEventListener("change", () => renderMap(DATA.trajectories[currentIndex]));
      $("#horizontal-rating").addEventListener("change", (event) => updateRecord((record) => { record.horizontal = event.target.value; }));
      $("#altitude-rating").addEventListener("change", (event) => updateRecord((record) => { record.altitude = event.target.value; }));
      $("#speed-rating").addEventListener("change", (event) => updateRecord((record) => { record.speed = event.target.value; }));
      $("#notes").addEventListener("input", (event) => updateRecord((record) => { record.notes = event.target.value; }));
      root.querySelectorAll("[data-decision]").forEach((button) => button.addEventListener("click", () => {
        updateRecord((record) => { record.decision = button.dataset.decision; });
        loadAssessment(DATA.trajectories[currentIndex]);
      }));
      $("#export-csv").addEventListener("click", exportCsv);
      document.addEventListener("keydown", (event) => {
        if ($("#detail-panel").classList.contains("hidden") || event.target.matches("textarea, select, input")) return;
        const decision = {k: "keep", r: "review", x: "reject"}[event.key.toLowerCase()];
        if (decision) {
          updateRecord((record) => { record.decision = decision; });
          loadAssessment(DATA.trajectories[currentIndex]);
        }
      });
      updateSummary(); renderOverview();
    })();
  </script>
</div>
"""


def write_bundle(payload: dict[str, object], output: Path) -> Path:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__DATA__", data)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a dependency-free browser review bundle."
    )
    parser.add_argument(
        "--generated",
        type=Path,
        default=(
            PROJECT_ROOT
            / "runs"
            / RUN_NAME
            / "samples"
            / SAMPLE_NAME
            / "generated.parquet"
        ),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / ARTIFACT_NAME,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "runs"
            / RUN_NAME
            / "reviews"
            / SAMPLE_NAME
            / "manual-v1"
            / "index.html"
        ),
    )
    args = parser.parse_args()
    if not args.generated.exists():
        raise FileNotFoundError(args.generated)
    if not args.artifact.exists():
        raise FileNotFoundError(args.artifact)
    output = write_bundle(build_payload(args.generated, args.artifact), args.output)
    print(f"OUTPUT={output.resolve()}")


if __name__ == "__main__":
    main()
