"""Save the selected model's actual checkpoint configuration."""

import json
import sys
from pathlib import Path

import torch

checkpoint = Path(sys.argv[1])
output = Path(sys.argv[2])
if output.exists():
    raise FileExistsError(output)
state = torch.load(checkpoint, map_location="cpu", weights_only=False)
config = state["hyper_parameters"]["resolved_config"]
result = {
    "epoch": int(state["epoch"]),
    "resolved_config": config,
    "prior": config["model"].get("prior", {"type": "standard"}),
}
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
print(json.dumps(result, indent=2))
print(f"OUTPUT={output.resolve()}")
