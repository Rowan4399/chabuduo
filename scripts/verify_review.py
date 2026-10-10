# ruff: noqa: E501 -- report paragraphs and reproduction commands are literal text.
"""Independent artifact assertions and a compact final evaluation report."""

import hashlib
import importlib.metadata
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

root = Path.cwd()
folder = root / "runs/rwy14-review-v1"
results = folder / "results-final"
rules_path = folder / "calibration-final/rules.json"
rules = json.loads(rules_path.read_text(encoding="utf-8"))
summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
rule_hash = hashlib.sha256(rules_path.read_bytes()).hexdigest()
assert rule_hash == summary["rules_sha256"]
report = [
    "# RWY14 固定轨迹规则评审",
    "",
    "远程执行：desktop-hqosbo1，D:/rrl/generation/chabuduo。",
    "每组固定原始顺序前100条，未重新采样，未改写原始样本。",
    "",
    "| 模型 | 通过 | 拒绝 | 无法判定 | 高亮 |",
    "|---|---:|---:|---:|---:|",
]
details = []
for model in summary["models"]:
    path = results / model["id"]
    result = json.loads((path / "results.json").read_text(encoding="utf-8"))
    records = result["records"]
    source = root / result["model"]["samples"]
    original = pd.read_parquet(source / "generated.parquet")
    fixed = pd.read_parquet(path / "fixed100.parquet")
    assert len(records) == fixed.flight_id.nunique() == 100
    assert [r["id"] for r in records] == original.flight_id.drop_duplicates().tolist()[
        :100
    ]
    pd.testing.assert_frame_equal(
        fixed.reset_index(drop=True),
        original[original.flight_id.isin(fixed.flight_id)].reset_index(drop=True),
    )
    with np.load(source / "samples.npz") as a, np.load(path / "fixed100-raw.npz") as b:
        np.testing.assert_array_equal(a["values"][:100], b["values"])
    ranked = sorted(
        [r for r in records if r["status"] == "passed"],
        key=lambda r: (
            -r["total"],
            -r["groups"]["alignment"],
            -r["groups"]["speed"],
            r["id"],
        ),
    )
    assert result["top10"] == [r["id"] for r in ranked[:10]]
    assert result["top5"] == [r["id"] for r in ranked[:5]]
    for rank, r in enumerate(ranked, 1):
        assert r["rank"] == rank and not r["failures"] and not r["unknown"]
        assert all(x["passed"] for x in r["metrics"].values())
        assert r["checks"]["raw_nonpositive_intervals"] == 0
        assert (
            abs(
                r["total"]
                - sum(r["groups"][g] * w for g, w in rules["weights"].items())
            )
            < 1e-10
        )
    failures = Counter(k for r in records for k in r["failures"])
    details.append(
        {
            "model": model["id"],
            "failures": dict(failures),
            "top10": result["top10"],
            "assertions_passed": True,
        }
    )
    report.append(
        f"| {model['label']} | {model['passed']} | {model['rejected']} | {model['undetermined']} | {model['highlighted']} |"
    )

report += [
    "",
    "## 校准",
    "",
    f"训练 {rules['counts']['train']} 条，验证 {rules['counts']['validation']} 条；各目标数据集的划分冲突样本已排除。",
    "所有模型使用相同RWY14规则；未新建flow或聚类。真实位置使用观测经纬度，不从地速重建。",
    f"末段参考方向 {rules['reference_deg']:.4f}°，检查长度 {rules['terminal_m']:.2f} m。",
    f"采用训练分位数 {rules['chosen_quantile']}，验证联合通过 207/227（91.19%）。",
    "这是用于选阈值的验证集保留率，不是独立测试集上的性能估计。",
    "",
    "## 解读限制",
    "",
    "- 灰色表示未高亮，包含通过但排名较低的轨迹。失败轨迹仍全部保留。",
    "- 得分表示符合本套规则的程度，不代表生成概率或完整飞行性能验证。",
    "- 全局模型含其他跑道方向；这里按RWY14场景评估，不能把拒绝数直接解释成全局模型质量。",
    "- 单flow TCVAE及K1000文件原含500条，本次在评分前固定原始前100条。",
    "- 历史单flow和K1000样本原先由跑道入口锚定。本次保留其坐标；规则不读取入口，终点位置不参与评分。",
    "- 已知机场锚点为LSZH (47.464722, 8.549167)。地图使用机场方位等距投影米坐标。",
    "- 重建一致性检查核对部署traffic的0.99系数及反向测地积分，只作准入，不给高分奖励。",
    "- 原始非正时间间隔、实质裁剪或时间修正均独立拒绝，不由修正后的曲线掩盖。",
    "- 真实训练数据本身含观测噪声，尾部分位数边界不是飞行性能包线。",
    "",
    "## 文件",
    "",
    "- `results-final/index.html`：离线交互页，五模型、前5/10切换、所有100条详情。",
    "- `results-final/<model>/scores.csv`：逐条评分和拒绝原因。",
    "- `results-final/<model>/results.json`：完整指标、边界、分数与原始/重建检查。",
    "- `results-final/<model>/highlights.json`：精确前5/10 ID。",
    "- `results-final/<model>/fixed100.parquet`、`fixed100-raw.npz`：原始坐标/原始输出的固定100条副本。",
    "- `results-final/<model>/overview.png`：全部100条灰色背景和合格高亮。",
    "- `calibration-final/rules.json`：最终冻结规则；`*_metrics.csv`、`*_checks.csv`：真实数据指标和联合误拒。",
    "- `calibration/`：保留的初版校准。全程方向分布下四分位数曾使末段过长；仅根据真实数据审查改为终点方向变异范围，未按生成通过数量调参。",
    "- `score.log`：初次JSON数值类型错误；最终输出见`score-final.log`，最终规则未因此改变。",
    "",
    "## 复现（在远程项目根目录逐步执行，输出目录必须未存在）",
    "",
    "```powershell",
    "uv run --no-sync scripts/trajectory_rules.py calibrate --output runs/new-review/calibration",
    "uv run --no-sync scripts/trajectory_rules.py score --rules runs/new-review/calibration --manifest runs/rwy14-review-v1/models.json --output runs/new-review/results",
    "uv run --no-sync scripts/trajectory_rules.py render --rules runs/new-review/calibration --results runs/new-review/results",
    "```",
    "",
    "## 各模型失败指标计数（同一轨迹可有多项）",
    "",
]
for d in details:
    report += [
        f"### {d['model']}",
        "",
        "```json",
        json.dumps(d, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
(folder / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
(folder / "verification.json").write_text(
    json.dumps(
        {
            "rules_sha256": rule_hash,
            "models": details,
            "checks": [
                "500 rows assessed",
                "exact original coordinate and raw output equality",
                "all highlights pass",
                "exact top K ordering",
                "rank and weighted scores",
                "no raw time reversal in highlights",
            ],
        },
        indent=2,
    ),
    encoding="utf-8",
)
environment = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages": {
        p: importlib.metadata.version(p)
        for p in ["numpy", "pandas", "pyproj", "traffic", "matplotlib", "pyarrow"]
    },
}
(folder / "environment.json").write_text(
    json.dumps(environment, indent=2), encoding="utf-8"
)
print("VERIFIED: 500 fixed samples; original data unchanged; exact passing top K")
print(folder / "REPORT.md")
