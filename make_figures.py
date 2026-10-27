"""Make the report figures and tables from runs/*/ (metrics.json, history.csv, classification_report.txt).

    python make_figures.py            # writes figures/*.pdf, figures/*.png and per_class_f1.csv
"""
import csv
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = "runs"
OUT = "figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order
REF_GREY = "#8a8985"
INK, INK2 = "#0b0b0b", "#52514e"

ORDER = ["ref", "A_cnn", "A_gru", "B_conv1", "B_conv2", "C_w20", "C_w60",
         "D_full", "D_nodrop", "D_nowd", "D_noaug", "E_lr3e-3", "E_lr3e-3_warmup", "F_focal"]
CLASSES = ["Null", "Open Door 1", "Open Door 2", "Close Door 1", "Close Door 2", "Open Fridge",
           "Close Fridge", "Open Dishwasher", "Close Dishwasher", "Open Drawer 1", "Close Drawer 1",
           "Open Drawer 2", "Close Drawer 2", "Open Drawer 3", "Close Drawer 3", "Clean Table",
           "Drink from Cup", "Toggle Switch"]

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 1.6, "savefig.bbox": "tight",
})


def run_dir(cfg):
    return os.path.join(RUNS, f"{cfg}_s0")


def metrics(cfg):
    with open(os.path.join(run_dir(cfg), "metrics.json")) as f:
        return json.load(f)


def history(cfg):
    with open(os.path.join(run_dir(cfg), "history.csv")) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def per_class_f1(cfg):
    """Parse the per-class F1 column of sklearn's classification_report.txt."""
    f1 = {}
    with open(os.path.join(run_dir(cfg), "classification_report.txt")) as f:
        for line in f:
            m = re.match(r"\s*(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", line)
            if m and m.group(1) in CLASSES:
                f1[m.group(1)] = float(m.group(4))
    return [f1[c] for c in CLASSES]


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200)
    plt.close(fig)


def fig_overview():
    """Test weighted and macro F1 for every run; dashed line = reference."""
    runs = ORDER[::-1]
    ref = metrics("ref")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=True)
    for ax, key, title, lim in [(axes[0], "test_f1_weighted", "Weighted F1", (0.7, 0.92)),
                                (axes[1], "test_f1_macro", "Macro F1", (0.0, 0.7))]:
        vals = [metrics(r)[key] for r in runs]
        colors = [REF_GREY if r == "ref" else SERIES[0] for r in runs]
        ax.barh(runs, [v - lim[0] for v in vals], left=lim[0], height=0.62, color=colors)
        ax.axvline(ref[key], color=INK2, ls="--", lw=0.9)
        for y, v in enumerate(vals):
            ax.text(v + (lim[1] - lim[0]) * 0.01, y, f"{v:.3f}", va="center", fontsize=7.5, color=INK2)
        ax.set_xlim(*lim)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_xlabel("test score (axis starts at 0.70)")
    axes[1].set_xlabel("test score")
    fig.tight_layout()
    save(fig, "overview_f1")


def fig_per_class(cfgs, labels, name):
    """Grouped horizontal bars of per-class test F1 for a few runs."""
    data = [per_class_f1(c) for c in cfgs]
    y = np.arange(len(CLASSES))[::-1]
    h = 0.8 / len(cfgs)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for i, (d, lab) in enumerate(zip(data, labels)):
        ax.barh(y + (len(cfgs) - 1) / 2 * h - i * h, d, height=h * 0.9, color=SERIES[i], label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels(CLASSES)
    ax.set_xlim(0, 1)
    ax.set_xlabel("test F1")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncol=1, fontsize=8)
    fig.tight_layout()
    save(fig, name)


def fig_curves(cfgs, labels, name, panels):
    """One panel per history column; one line per run."""
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 2.7))
    for ax, (col, ylabel) in zip(np.atleast_1d(axes), panels):
        for i, (c, lab) in enumerate(zip(cfgs, labels)):
            h = history(c)
            ax.plot(h["epoch"], h[col], color=SERIES[i], label=lab)
            b = metrics(c)["best_epoch"]
            ax.plot(b, h[col][b - 1], "o", ms=5, color=SERIES[i], mec="white", mew=1.2)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
    np.atleast_1d(axes)[0].legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, name)


def per_class_table():
    with open("per_class_f1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class"] + ORDER)
        cols = {r: per_class_f1(r) for r in ORDER}
        for i, c in enumerate(CLASSES):
            w.writerow([c] + [f"{cols[r][i]:.3f}" for r in ORDER])


if __name__ == "__main__":
    fig_overview()
    fig_per_class(["ref", "A_cnn", "A_gru"], ["ref (LSTM)", "A_cnn (dense)", "A_gru (GRU)"], "perclass_A")
    fig_per_class(["ref", "B_conv1", "B_conv2"], ["ref (4 conv)", "B_conv1", "B_conv2"], "perclass_B")
    fig_per_class(["C_w20", "ref", "C_w60"], ["C_w20 (0.67 s)", "ref (0.8 s)", "C_w60 (2.0 s)"], "perclass_C")
    fig_per_class(["ref", "F_focal"], ["ref (cross-entropy)", "F_focal (focal, $\\gamma$=2)"], "perclass_F")
    fig_curves(["D_full", "D_nodrop", "D_nowd", "D_noaug"], ["D_full", "D_nodrop", "D_nowd", "D_noaug"],
               "curves_D", [("train_loss", "train loss"), ("val_loss", "val loss (CE)"),
                            ("val_f1_macro", "val macro F1")])
    fig_curves(["ref", "E_lr3e-3", "E_lr3e-3_warmup"], ["ref (lr 1e-3)", "lr 3e-3", "lr 3e-3 + warm-up"],
               "curves_E", [("lr", "learning rate"), ("train_loss", "train loss"),
                            ("val_f1_macro", "val macro F1")])
    fig_curves(["ref", "F_focal"], ["ref", "F_focal"], "curves_ref",
               [("train_loss", "train loss"), ("val_loss", "val loss (CE)"), ("val_f1_macro", "val macro F1")])
    per_class_table()
    print("figures written to", OUT)
