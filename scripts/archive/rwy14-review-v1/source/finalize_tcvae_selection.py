"""Apply the user's K256 single-flow and K1000 all-flow model selection."""

import json
import shutil
from pathlib import Path

run = Path("runs/rwy14-review-v1")
results = run / "results-final"
summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
addition = json.loads(
    (run / "k256-correction/summary.json").read_text(encoding="utf-8")
)
assert summary["rules_sha256"] == addition["rules_sha256"]
models = {m["id"]: m for m in summary["models"] + addition["models"]}
specs = {
    m["id"]: m for m in summary["manifest"]["models"] + addition["manifest"]["models"]
}
ids = [
    "ddpm-flow700",
    "ddpm-global700",
    "tcvae-flow-k256",
    "tcvae-k1000",
    "latent-k1000",
]
shutil.copytree(run / "k256-correction/tcvae-flow-k256", results / "tcvae-flow-k256")
label = "TCVAE 单flow · VampPrior K256 · best epoch 298"
models["tcvae-flow-k256"]["label"] = label
specs["tcvae-flow-k256"]["label"] = label
record_path = results / "tcvae-flow-k256/results.json"
record = json.loads(record_path.read_text(encoding="utf-8"))
record["model"]["label"] = label
record_path.write_text(
    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
)
manifest = {
    "scope": "User-confirmed five models: DDPM single/global at 700 epochs, "
    "TCVAE single-flow VampPrior K256, TCVAE RWY14 all-flow VampPrior K1000, "
    "Latent DDPM K1000. Same frozen rules and original first 100 per model.",
    "models": [specs[k] for k in ids],
}
summary["models"] = [models[k] for k in ids]
summary["manifest"] = manifest
(results / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
shutil.copy2(run / "models.json", run / "models-before-k1000.json")
for name in ["models.json", "models-current.json"]:
    (run / name).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
print(results / "summary.json")
