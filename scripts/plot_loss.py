from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent

log_dir = (
    project_dir
    / "runs"
    / "beijing_tcvae-v1-seed-42"
    / "logs"
    / "training"
)

files = sorted(log_dir.glob("version_*/metrics.csv"))

if not files:
    raise FileNotFoundError(f"No metrics.csv found in {log_dir}")

metrics = pd.concat(
    [pd.read_csv(path) for path in files],
    ignore_index=True,
)

train_loss = (
    metrics.dropna(subset=["train/loss_epoch"])
    .groupby("epoch")["train/loss_epoch"]
    .last()
)

validation_loss = (
    metrics.dropna(subset=["validation/loss"])
    .groupby("epoch")["validation/loss"]
    .last()
)


def save_plot(filename: str, logarithmic: bool = False) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))

    axis.plot(train_loss, label="Train loss")
    axis.plot(validation_loss, label="Validation loss")

    if logarithmic:
        minimum = min(train_loss.min(), validation_loss.min())
        axis.set_yscale("log" if minimum > 0 else "symlog")
        axis.set_title("TCVAE Loss — Log Scale")
    else:
        axis.set_title("TCVAE Loss")

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(script_dir / filename, dpi=200)
    plt.close(figure)


save_plot("loss_curves.png")
save_plot("loss_curves_log.png", logarithmic=True)