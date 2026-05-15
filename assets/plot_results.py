"""Generate the comparison plots used in README.md.

Re-runnable: parses the two server-side training logs and the saved
comparison_returns.npz, writes two PNGs under assets/.

Usage:
    python3 assets/plot_results.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ASSETS = Path(__file__).resolve().parent
LOGS = ASSETS / "data"


def parse_vector_evals(path: Path):
    """Vector log lines:
        Evaluation at step 50000:
        Mean episode reward: 9.14 +/- 0.17
    Returns (steps, means, stds).
    """
    text = path.read_text()
    step_re = re.compile(r"Evaluation at step (\d+):")
    val_re = re.compile(r"Mean episode reward: ([\d\.\-]+) \+/- ([\d\.\-]+)")
    steps, means, stds = [], [], []
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        m = step_re.search(ln)
        if not m:
            continue
        for j in range(i + 1, min(i + 4, len(lines))):
            v = val_re.search(lines[j])
            if v:
                steps.append(int(m.group(1)))
                means.append(float(v.group(1)))
                stds.append(float(v.group(2)))
                break
    return np.array(steps), np.array(means), np.array(stds)


def parse_visual_evals(path: Path):
    """Visual log lines:
        [eval ] step 100,000 | return 9.00 +/- 0.00
    Returns (steps, means, stds).
    """
    text = path.read_text()
    pat = re.compile(r"\[eval \] step ([\d,]+) \| return ([\d\.\-]+) \+/- ([\d\.\-]+)")
    steps, means, stds = [], [], []
    for m in pat.finditer(text):
        steps.append(int(m.group(1).replace(",", "")))
        means.append(float(m.group(2)))
        stds.append(float(m.group(3)))
    return np.array(steps), np.array(means), np.array(stds)


def plot_learning_curves(out: Path) -> None:
    vec_s, vec_m, vec_sd = parse_vector_evals(LOGS / "vector_train.log")
    vis_s, vis_m, vis_sd = parse_visual_evals(LOGS / "visual_train.log")
    print(f"vector evals: {len(vec_s)}")
    print(f"visual evals: {len(vis_s)}")

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=150)

    c_vec, c_vis = "#1f77b4", "#d62728"
    ax.plot(vec_s, vec_m, "-o", color=c_vec, label="Vector DQN (FlappyBird-v0, 12-D obs)", markersize=5)
    ax.fill_between(vec_s, vec_m - vec_sd, vec_m + vec_sd, color=c_vec, alpha=0.15)
    ax.plot(vis_s, vis_m, "-s", color=c_vis, label="Visual DQN (FlappyBird-rgb-v0, pixels)", markersize=5)
    ax.fill_between(vis_s, vis_m - vis_sd, vis_m + vis_sd, color=c_vis, alpha=0.15)

    # mark each agent's best checkpoint
    i_vec = int(np.argmax(vec_m)); i_vis = int(np.argmax(vis_m))
    ax.scatter([vec_s[i_vec]], [vec_m[i_vec]], s=120, facecolor="none",
               edgecolor=c_vec, linewidth=2, zorder=5)
    ax.scatter([vis_s[i_vis]], [vis_m[i_vis]], s=120, facecolor="none",
               edgecolor=c_vis, linewidth=2, zorder=5)
    ax.annotate(f"best {vec_m[i_vec]:.0f}", (vec_s[i_vec], vec_m[i_vec]),
                xytext=(-15, 12), textcoords="offset points", color=c_vec, fontsize=9)
    ax.annotate(f"best {vis_m[i_vis]:.1f}", (vis_s[i_vis], vis_m[i_vis]),
                xytext=(-15, 12), textcoords="offset points", color=c_vis, fontsize=9)

    ax.set_yscale("log")
    ax.set_xlabel("Training env steps")
    ax.set_ylabel("Eval mean episode return (log scale)")
    ax.set_title("Eval return during training — Vector DQN vs Visual DQN on Flappy Bird")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


def plot_fair_comparison(out: Path) -> None:
    d = np.load(LOGS / "comparison_returns.npz")
    vec = d["vector_returns"]
    vis = d["visual_returns"]
    n = len(d["eval_seeds"])

    fig, (ax_box, ax_pair) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150,
                                          gridspec_kw={"width_ratios": [1, 1.4]})
    c_vec, c_vis = "#1f77b4", "#d62728"

    # ---- Left: box plot ----
    bp = ax_box.boxplot(
        [vec, vis],
        tick_labels=["Vector DQN", "Visual DQN"],
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, c in zip(bp["boxes"], [c_vec, c_vis]):
        patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(c)
    # Overlay individual points
    rng = np.random.default_rng(0)
    for i, (data, c) in enumerate([(vec, c_vec), (vis, c_vis)], start=1):
        xs = rng.normal(i, 0.05, size=len(data))
        ax_box.scatter(xs, data, color=c, alpha=0.6, s=14, zorder=3)
    ax_box.set_yscale("log")
    ax_box.set_ylabel("Episode return (log scale)")
    ax_box.set_title(f"Final eval: {n} episodes per agent\n(best checkpoint, ε=0, shared seeds)")
    ax_box.grid(True, axis="y", which="both", linestyle=":", alpha=0.4)
    ax_box.annotate(
        f"mean {vec.mean():.1f}", xy=(1, vec.mean()),
        xytext=(8, 0), textcoords="offset points", color=c_vec, fontsize=9, va="center",
    )
    ax_box.annotate(
        f"mean {vis.mean():.1f}", xy=(2, vis.mean()),
        xytext=(8, 0), textcoords="offset points", color=c_vis, fontsize=9, va="center",
    )

    # ---- Right: per-seed paired bar chart (sorted by vector return desc) ----
    order = np.argsort(-vec)
    idx = np.arange(n)
    width = 0.42
    ax_pair.bar(idx - width / 2, vec[order], width=width, color=c_vec, alpha=0.85, label="Vector DQN")
    ax_pair.bar(idx + width / 2, vis[order], width=width, color=c_vis, alpha=0.85, label="Visual DQN")
    ax_pair.set_yscale("log")
    ax_pair.set_xlabel(f"Eval episode index (sorted by Vector's return, {n} shared seeds)")
    ax_pair.set_ylabel("Episode return (log scale)")
    ax_pair.set_title(f"Per-seed paired comparison — Vector wins {(vec > vis).sum()}/{n} episodes")
    ax_pair.grid(True, axis="y", which="both", linestyle=":", alpha=0.4)
    ax_pair.legend(loc="upper right", framealpha=0.95)
    ax_pair.set_xticks([])

    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_learning_curves(ASSETS / "learning_curves.png")
    plot_fair_comparison(ASSETS / "fair_comparison.png")
