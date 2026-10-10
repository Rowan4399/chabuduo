# ruff: noqa: E501 -- literal report paragraphs
"""Correct the selected TCVAE model without recalibrating or sampling."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

root = Path.cwd()
run = root / "runs/rwy14-review-v1"
results = run / "results-final"
addition = run / "k1000-correction"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


old = read(results / "summary.json")
new = read(addition / "summary.json")
assert old["rules_sha256"] == new["rules_sha256"]
assert "tcvae-k1000" not in [m["id"] for m in old["models"]]
for name in ["summary.json", "index.html"]:
    original = results / name
    backup = results / (original.stem + "-before-k1000" + original.suffix)
    if backup.exists():
        raise FileExistsError(backup)
    shutil.copy2(original, backup)
for name in ["REPORT.md", "verification.json"]:
    original = run / name
    shutil.copy2(original, run / (original.stem + "-before-k1000" + original.suffix))

shutil.copytree(addition / "tcvae-k1000", results / "tcvae-k1000")
selected_ids = ["ddpm-flow700", "ddpm-global700", "tcvae-k1000", "latent-k1000"]
models = {m["id"]: m for m in old["models"] + new["models"]}
specs = {m["id"]: m for m in old["manifest"]["models"] + new["manifest"]["models"]}
manifest = {
    "scope": "Corrected TCVAE to RWY14 all-flow VampPrior K1000. Four confirmed "
    "distinct models; no single-flow K1000 run exists in this project. "
    "Old standard-prior TCVAE results retained as historical artifacts.",
    "models": [specs[k] for k in selected_ids],
}
summary = {
    "rules_sha256": old["rules_sha256"],
    "models": [models[k] for k in selected_ids],
    "manifest": manifest,
}
write(results / "summary.json", summary)
write(run / "models-current.json", manifest)

verified = []
for model in summary["models"]:
    folder = results / model["id"]
    result = read(folder / "results.json")
    assert result["rules_sha256"] == summary["rules_sha256"]
    source = root / result["model"]["samples"]
    original = pd.read_parquet(source / "generated.parquet")
    fixed = pd.read_parquet(folder / "fixed100.parquet")
    ids = original.flight_id.drop_duplicates().tolist()[:100]
    assert ids == [r["id"] for r in result["records"]]
    pd.testing.assert_frame_equal(
        original[original.flight_id.isin(ids)].reset_index(drop=True),
        fixed.reset_index(drop=True),
    )
    with np.load(source / "samples.npz") as a:
        with np.load(folder / "fixed100-raw.npz") as b:
            np.testing.assert_array_equal(a["values"][:100], b["values"])
    ranked = sorted(
        [r for r in result["records"] if r["status"] == "passed"],
        key=lambda r: (
            -r["total"],
            -r["groups"]["alignment"],
            -r["groups"]["speed"],
            r["id"],
        ),
    )
    assert result["top10"] == [r["id"] for r in ranked[:10]]
    assert result["top5"] == [r["id"] for r in ranked[:5]]
    assert all(not r["failures"] and not r["unknown"] for r in ranked)
    verified.append(model["id"])
rule_hash = hashlib.sha256(
    (run / "calibration-final/rules.json").read_bytes()
).hexdigest()
assert rule_hash == summary["rules_sha256"]
write(
    run / "verification.json",
    {
        "models": verified,
        "count": 400,
        "rules_sha256": rule_hash,
        "checks": [
            "original rows and coordinates unchanged",
            "raw NPZ unchanged",
            "exact passing top5/top10",
            "same frozen rules as original review",
        ],
    },
)

lines = [
    "# RWY14 轨迹评审：TCVAE K1000 更正",
    "",
    "TCVAE 已改为 VampPrior K1000，RWY14 全flow，best epoch 240。",
    "远程未找到单flow K1000；当前展示4个已确认的独立模型，未将同一模型重复标为两种模型。",
    "原标准先验 TCVAE 两项已移出主页面，历史数据及更正前页面仍保留。",
    "",
    "| 模型（各100条） | 通过 | 拒绝 | 高亮 |",
    "|---|---:|---:|---:|",
]
for m in summary["models"]:
    lines.append(
        f"| {m['label']} | {m['passed']} | {m['rejected']} | {m['highlighted']} |"
    )
lines += [
    "",
    "继续使用原冻结规则，哈希：`" + rule_hash + "`。",
    "训练6237条，验证227条，验证保留207条（91.19%）；未重校准、未重新采样。",
    "K1000在已有500条中按原始顺序固定前100条。全部坐标保持原样，规则不读取跑道入口。",
    "通过和得分仅表示符合本套RWY14规则，不是完整飞行性能验证。",
    "",
    "当前页面：`results-final/index.html`；当前模型清单：`models-current.json`。",
    "K1000详细结果：`results-final/tcvae-k1000/`。",
    "历史报告：`REPORT-before-k1000.md`；历史页面：`results-final/index-before-k1000.html`。",
    "原历史报告中的TCVAE 3/100、0/100通过率属于标准先验模型，不适用于K1000。",
]
(run / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("VERIFIED corrected model selection: 400 unchanged samples; frozen rules")
print(run / "REPORT.md")
