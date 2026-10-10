"""Verify the display update against the frozen review artifacts and package it."""

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

root = Path.cwd()
base = root / "runs/rwy14-review-v1"
results = base / "results-final"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


baseline = json.loads((base / "checksums.json").read_text(encoding="utf-8"))
summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
unchanged = ["calibration-final/rules.json", "results-final/summary.json"]
for model in summary["models"]:
    for name in [
        "results.json",
        "scores.csv",
        "highlights.json",
        "fixed100.parquet",
        "fixed100-raw.npz",
    ]:
        unchanged.append(f"results-final/{model['id']}/{name}")
for name in unchanged:
    assert sha(base / name) == baseline[name], name

display = json.loads((results / "training-display.json").read_text(encoding="utf-8"))
with np.load(base / "calibration-final/real-observed-200.npz") as cache:
    ids, split, xy, flow = (cache[k] for k in ["ids", "split", "xy", "flow"])
lookup = {str(ident): i for i, ident in enumerate(ids)}
for cohort, data in display["cohorts"].items():
    assert len({row["id"] for row in data["records"]}) == 100
    for row in data["records"]:
        i = lookup[row["id"]]
        assert split[i] == "train"
        np.testing.assert_equal(np.column_stack([row["x"], row["y"]]), xy[i])
        if cohort == "single":
            assert flow[i] == "162-216"

audit = json.loads((results / "turn-review.json").read_text(encoding="utf-8"))
table = "\n".join(
    f"| {r['id']} | {r['turn_rule_failures']} | {r['alignment_failures']} | "
    f"{r['entry_radius_failures']} | {len(r['entry_radius_only'])} |"
    for r in audit
)
report = f"""# 训练对照与人工查看更新

执行位置：desktop-hqosbo1，D:/rrl/generation/chabuduo。
页面：results-final/index.html。旧页面保存在 index-before-training-compare.html。

- 左侧固定100条真实RWY14训练参考；右侧保留原来的100条生成轨迹。
- 单flow从1499条训练参考抽100条；全flow从6237条抽100条。先按ID排序，seed=42，无放回。
- 同一范围的模型共用同一组训练ID。DDPM全局的左侧为RWY14子集，不代表其全部跑道训练分布。
- 训练位置来自既有观测缓存，只平移终点；生成位置、评分、排名、规则哈希均不变。
- 两图及对应速度、高度剖面共用轴范围。
  地图、剖面、排名、全部列表、下拉框和规则复核ID均支持点击查看。
- 蓝色独立人工查看不改变通过/拒绝结论；橙色仍为通过者前K；正在查看的前K仍用红色。
- 训练、生成列表可按累计绝对转向量排序。这是查看顺序，不是盘旋识别，也不参与评分。

## 当前规则的影响

| 模型 | 转向规则失败 | 末段对准失败 | 直线距离失败 | 仅直线距离失败 |
|---|---:|---:|---:|---:|
{table}

各列可以重叠。拒绝数不能解释为不合理盘旋数。当前数据中，覆盖距离和末段对准影响更广；
这不证明所有被它们拒绝的轨迹都应该放行。v1不直接惩罚自交、盘旋或路径长，
但起终点直线距离门槛与转向时间指标仍可能拒绝可取轨迹。

本次未重新校准或放宽物理阈值。下一次规则修订应首先用真实训练、验证数据检查：
原始40nm选取过程是否确实要求生成起点远离终点，以及真实盘旋事件的转向率与持续时间分布。
不能仅根据累计转向量判定盘旋，也不能为了让生成样本通过而放宽阈值。

## 验证与复现

Ruff通过，8项针对性测试通过。真实显示样本的train归属、单flow归属和观测坐标逐条核对。
27个冻结规则、样本、评分与排名文件与此前交付的checksums.json一致。
本次文件哈希另存training-compare-checksums.json；此前checksums.json是旧版页面的历史基线。

```powershell
uv run --no-sync scripts/trajectory_rules.py render `
  --rules runs/rwy14-review-v1/calibration-final `
  --results runs/rwy14-review-v1/results-final
```
"""
(base / "TRAINING_COMPARE.md").write_text(report, encoding="utf-8")
verification = {
    "unchanged_count": len(unchanged),
    "unchanged": unchanged,
    "training_cohorts_verified": ["single", "global"],
    "seed": 42,
    "coordinate_changes": False,
    "rule_changes": False,
}
(base / "training-compare-verification.json").write_text(
    json.dumps(verification, indent=2), encoding="utf-8"
)
files = [
    root / "scripts/trajectory_rules.py",
    root / "scripts/trajectory_rules.html",
    root / "tests/test_trajectory_rules.py",
    Path(__file__),
    results / "index.html",
    results / "training-display.json",
    results / "turn-review.json",
    base / "TRAINING_COMPARE.md",
    base / "training-compare-verification.json",
]
checksums = {p.relative_to(root).as_posix(): sha(p) for p in files}
checksum_file = base / "training-compare-checksums.json"
checksum_file.write_text(json.dumps(checksums, indent=2), encoding="utf-8")
archive = base / "training-compare-update.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for path in files + [checksum_file]:
        z.write(path, path.relative_to(root).as_posix())
print(json.dumps(verification))
print(archive)
