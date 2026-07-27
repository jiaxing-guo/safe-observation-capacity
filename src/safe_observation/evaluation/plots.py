"""Plots primitives for safe-observation experiments. See supplementary Reproducibility for its role in the release workflow."""

from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Plotting requires matplotlib. Install the viz extra: "
        "`uv sync --extra viz` (or add it to your sync extras)."
    ) from exc


def _rounds(results: dict[str, Any]) -> list[int]:
    """Compute rounds for the plots workflow."""
    return [row["round"] for row in results["aggregated_rounds"]]


def _series(results: dict[str, Any], key: str) -> tuple[list[float], list[float]]:
    """Compute series for the plots workflow."""
    rows = results["aggregated_rounds"]
    return [r[f"{key}_mean"] for r in rows], [r[f"{key}_std"] for r in rows]


def _band(ax, xs, mean_ys, std_ys, label, color=None):
    """Compute or draw a mean-and-uncertainty band."""
    (line,) = ax.plot(xs, mean_ys, label=label, color=color)
    lo = [m - s for m, s in zip(mean_ys, std_ys, strict=True)]
    hi = [m + s for m, s in zip(mean_ys, std_ys, strict=True)]
    ax.fill_between(xs, lo, hi, alpha=0.2, color=line.get_color())
    return line


def plot_value_over_rounds(results: dict[str, Any], ax=None):
    """Plot value over rounds for the plots workflow."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    xs = _rounds(results)
    _band(ax, xs, *_series(results, "actual_value"), "actual value $x_t^\\top A y^*$")
    _band(ax, xs, *_series(results, "avg_payoff"), "realized payoff")
    ax.axhline(
        results["game_value"],
        ls="--",
        c="gray",
        lw=1,
        label=f"game value ${results['game_value']:.4g}$",
    )
    ax.axhline(results["safety_floor"], ls=":", c="red", lw=1, label="safety floor")
    ax.set_xlabel("round")
    ax.set_ylabel("value (player 1)")
    ax.set_title(f"Value over rounds vs {results['opponent']}")
    ax.legend(fontsize=8)
    return ax


def plot_safety_over_rounds(results: dict[str, Any], ax=None):
    """Plot safety over rounds for the plots workflow."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    xs = _rounds(results)
    _band(
        ax,
        xs,
        *_series(results, "safety_value"),
        "safety value $\\min_y x_t^\\top A y$",
    )
    ax.axhline(results["safety_floor"], ls=":", c="red", lw=1, label="safety floor")
    ax.set_xlabel("round")
    ax.set_ylabel("worst-case value")
    ax.set_title(f"Safety lower bound vs {results['opponent']}")
    ax.legend(fontsize=8)
    return ax


def plot_ci_shrinkage(results: dict[str, Any], ax=None):
    """Plot confidence interval shrinkage for the plots workflow."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    xs = _rounds(results)
    rows = results["aggregated_rounds"]
    for label in rows[0]["ci_width_by_infoset_mean"]:
        ax.plot(xs, [r["ci_width_by_infoset_mean"][label] for r in rows], label=label)
    ax.set_xlabel("round")
    ax.set_ylabel("mean CI width")
    ax.set_title(f"Confidence-interval shrinkage vs {results['opponent']}")
    ax.legend(fontsize=7, ncol=2)
    return ax


def plot_robust_vs_empirical_br(results: dict[str, Any], ax=None):
    """Plot robust vs empirical br."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    xs = _rounds(results)
    _band(ax, xs, *_series(results, "robust_value"), "robust value (safe)")
    _band(
        ax,
        xs,
        *_series(results, "empirical_br_value"),
        "empirical best response (unsafe)",
    )
    ax.axhline(
        results["game_value"],
        ls="--",
        c="gray",
        lw=1,
        label=f"game value ${results['game_value']:.4g}$",
    )
    ax.set_xlabel("round")
    ax.set_ylabel("value (player 1)")
    ax.set_title(f"Robust vs empirical-BR value vs {results['opponent']}")
    ax.legend(fontsize=8)
    return ax


def plot_ablation(ablation: dict[str, list[dict[str, Any]]], ax=None):
    """Plot ablation for the plots workflow."""
    axes = ax
    if axes is None:
        _fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for panel, axis_name in zip(axes, ("delta", "eps_safe", "method"), strict=True):
        cells = ablation[axis_name]
        xs = [str(c[axis_name]) for c in cells]
        gains = [c["exploitation_gain_mean"] for c in cells]
        errs = [c["exploitation_gain_std"] for c in cells]
        panel.bar(xs, gains, yerr=errs, capsize=3)
        panel.set_xlabel(axis_name)
        panel.set_ylabel("exploitation gain")
        panel.set_title(f"vs {axis_name}")
    return axes


def render_online_figures(
    results: dict[str, Any], out_dir: str | Path = "results"
) -> dict[str, Path]:
    """Render online figures for the plots workflow."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    opp = results["opponent"]
    panels = {
        "value": plot_value_over_rounds,
        "safety": plot_safety_over_rounds,
        "ci_shrinkage": plot_ci_shrinkage,
        "robust_vs_br": plot_robust_vs_empirical_br,
    }
    saved: dict[str, Path] = {}
    for name, fn in panels.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        fn(results, ax=ax)
        fig.tight_layout()
        path = out / f"online_{opp}_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved[name] = path
    return saved


def render_ablation_figure(
    ablation: dict[str, list[dict[str, Any]]],
    opponent: str,
    out_dir: str | Path = "results",
) -> Path:
    """Render ablation figure for the plots workflow."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    plot_ablation(ablation, ax=axes)
    fig.tight_layout()
    path = out / f"ablation_{opponent}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _arm_series(arm: dict[str, Any], key: str) -> tuple[list[float], list[float]]:
    """Compute arm series for the plots workflow."""
    rows = arm[key]
    return [r["mean"] for r in rows], [r["std"] for r in rows]


def plot_passive_vs_probing(comparison: dict[str, Any], axes=None):
    """Plot passive vs probing for the plots workflow."""
    if axes is None:
        _fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    arms = comparison["arms"]
    rounds = list(range(comparison["rounds"]))
    v = comparison["game_value"]

    for name, arm in arms.items():
        _band(axes[0], rounds, *_arm_series(arm, "actual_value_by_round"), name)
    axes[0].axhline(v, ls="--", c="gray", lw=1, label="game value")
    axes[0].set_xlabel("round")
    axes[0].set_ylabel("actual value $x_t^\\top A y^*$")
    axes[0].set_title("Exploitation: passive vs probing")
    axes[0].legend(fontsize=8)

    for name, arm in arms.items():
        _band(axes[1], rounds, *_arm_series(arm, "target_ci_by_round"), name)
    axes[1].set_xlabel("round")
    axes[1].set_ylabel("mean CI width at leak sites")
    axes[1].set_title(
        f"Leak-site identification ({comparison['n_target_labels']} sets)"
    )
    axes[1].legend(fontsize=8)

    for name, arm in arms.items():
        _band(axes[2], rounds, *_arm_series(arm, "budget_spent_by_round"), name)
    axes[2].set_xlabel("round")
    axes[2].set_ylabel("cumulative safety budget spent")
    axes[2].set_title("Probing budget")
    axes[2].legend(fontsize=8)
    return axes


def render_probing_figure(
    comparison: dict[str, Any], out_dir: str | Path = "results"
) -> Path:
    """Render probing figure for the plots workflow."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_passive_vs_probing(comparison, axes=axes)
    fig.tight_layout()
    path = out / f"probing_{comparison['opponent']}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_budget_frontier(frontier: dict[str, Any], axes=None):
    """Plot budget frontier for the plots workflow."""
    if axes is None:
        _fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cells = frontier["cells"]
    spent = [c["budget_spent_mean"] for c in cells]
    gain = [c["exploitation_gain_mean"] for c in cells]
    err = [c["exploitation_gain_std"] for c in cells]
    vpb = [c["value_per_budget"] for c in cells]

    axes[0].errorbar(
        spent, gain, yerr=err, marker="o", capsize=3, label="exploitation gain"
    )
    axes[0].axhline(
        frontier["scbr_gain"],
        ls="--",
        c="green",
        lw=1,
        label="SCBR ceiling (hard-safe)",
    )
    axes[0].axhline(0.0, ls=":", c="gray", lw=1, label="game value")
    axes[0].set_xlabel("safety budget spent (cumulative slack)")
    axes[0].set_ylabel("exploitation gain")
    axes[0].set_title(f"Safety-exploitation frontier vs {frontier['opponent']}")
    axes[0].legend(fontsize=8)

    axes[1].plot([c["per_round_budget"] for c in cells], vpb, marker="o")
    axes[1].set_xlabel("per-round budget $\\rho$")
    axes[1].set_ylabel("value above SCBR per unit budget")
    axes[1].set_title("Value per unit safety budget")
    return axes


def render_budget_frontier_figure(
    frontier: dict[str, Any], out_dir: str | Path = "results"
) -> Path:
    """Render budget frontier figure for the plots workflow."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_budget_frontier(frontier, axes=axes)
    fig.tight_layout()
    path = out / f"budget_frontier_{frontier['opponent']}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_importance_comparison(comparison: dict[str, Any], axes=None):
    """Plot importance comparison for the plots workflow."""
    if axes is None:
        _fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cells = comparison["cells"]
    rho = [c["per_round_budget"] for c in cells]
    uni = [c["uniform"]["exploitation_gain_mean"] for c in cells]
    uni_err = [c["uniform"]["exploitation_gain_std"] for c in cells]
    sen = [c["sensitivity"]["exploitation_gain_mean"] for c in cells]
    sen_err = [c["sensitivity"]["exploitation_gain_std"] for c in cells]
    adv = [c["sensitivity_advantage"] for c in cells]

    axes[0].errorbar(
        rho, uni, yerr=uni_err, marker="o", capsize=3, label="uniform (width)"
    )
    axes[0].errorbar(
        rho, sen, yerr=sen_err, marker="s", capsize=3, label="sensitivity (dual price)"
    )
    axes[0].axhline(
        comparison["scbr_gain"], ls="--", c="green", lw=1, label="SCBR ceiling"
    )
    axes[0].set_xlabel("per-round budget $\\rho$")
    axes[0].set_ylabel("exploitation gain")
    axes[0].set_title(f"Importance weighting vs {comparison['opponent']}")
    axes[0].legend(fontsize=8)

    axes[1].bar([str(r) for r in rho], adv)
    axes[1].axhline(0.0, c="gray", lw=1)
    axes[1].set_xlabel("per-round budget $\\rho$")
    axes[1].set_ylabel("sensitivity advantage (gain)")
    axes[1].set_title("Extra exploitation from sensitivity weighting")
    return axes


def render_importance_figure(
    comparison: dict[str, Any], out_dir: str | Path = "results"
) -> Path:
    """Render importance figure for the plots workflow."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_importance_comparison(comparison, axes=axes)
    fig.tight_layout()
    path = out / f"importance_{comparison['opponent']}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_coverage(coverage: dict[str, Any], ax=None):
    """Plot coverage for the plots workflow."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    cells = coverage["cells"]
    deltas = [c["delta"] for c in cells]
    for arm, marker in (("no_union", "x"), ("spatial", "o"), ("time_uniform", "s")):
        ys = [c["arms"][arm]["anytime_coverage"] for c in cells]
        ax.plot(deltas, ys, marker=marker, label=arm)
    ax.plot(
        deltas,
        [c["guarantee"] for c in cells],
        ls="--",
        c="gray",
        lw=1,
        label="$1-\\delta$ guarantee",
    )
    ax.set_xlabel("$\\delta$")
    ax.set_ylabel("anytime coverage  $\\Pr[y^* \\in C_t\\ \\forall t]$")
    ax.set_title(f"Confidence-set coverage vs {coverage['opponent']}")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    return ax


def render_coverage_figure(
    coverage: dict[str, Any], out_dir: str | Path = "results"
) -> Path:
    """Render coverage figure for the plots workflow."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_coverage(coverage, ax=ax)
    fig.tight_layout()
    path = out / f"coverage_{coverage['opponent']}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_finite_sample_gap(gap: dict[str, Any], axes=None):
    """Plot finite sample gap for the plots workflow."""
    if axes is None:
        _fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    series = gap["series"]
    rounds = [r["round"] for r in series]
    radii = [r["radius"] for r in series]
    gaps = [r["scbr_gap"] for r in series]

    axes[0].plot(rounds, gaps, marker="o", ms=3, label="SCBR gap")
    axes[0].plot(rounds, radii, marker="s", ms=3, label="confidence radius")
    axes[0].set_xlabel("round")
    axes[0].set_ylabel("value")
    axes[0].set_title(f"Gap & radius over rounds vs {gap['opponent']}")
    axes[0].legend(fontsize=8)

    c = gap["bound_constant"]
    axes[1].scatter(radii, gaps, s=18, label="rounds")
    xs = [0.0, max(radii)] if radii else [0.0, 1.0]
    axes[1].plot(
        xs,
        [c * x for x in xs],
        ls="--",
        c="red",
        lw=1,
        label=f"$C\\cdot$radius, $C={c:.2f}$",
    )
    axes[1].set_xlabel("confidence radius")
    axes[1].set_ylabel("SCBR gap")
    axes[1].set_title("Gap is $O(\\mathrm{radius})$")
    axes[1].legend(fontsize=8)
    return axes


def render_finite_sample_gap_figure(
    gap: dict[str, Any], out_dir: str | Path = "results"
) -> Path:
    """Render finite sample gap figure."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_finite_sample_gap(gap, axes=axes)
    fig.tight_layout()
    path = out / f"finite_gap_{gap['opponent']}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


_GUARANTEE_COLOR = {
    "hard_safe": "#c8e6c9",
    "certified_budget": "#fff9c4",
    "gift_funded": "#bbdefb",
    "none": "#ffcdd2",
}


def _render_table(ax, col_labels, rows, row_colors):
    """Render table for the plots workflow."""
    ax.axis("off")
    cell_colors = [[c] * len(col_labels) for c in row_colors]
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    for (r, _c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")
    return table


def plot_baseline_comparison(comparison: dict[str, Any], ax=None):
    """Plot baseline comparison for the plots workflow."""
    ax = ax or plt.subplots(figsize=(9, 3.5))[1]
    rows_data = comparison["opponents"]
    method_names = comparison["method_names"]
    first = next(iter(rows_data.values()))["methods"]

    col_labels = [
        "method",
        "guarantee",
        "mean gain",
        "worst violation",
        "mean dist. SCBR",
        "held",
    ]
    table_rows = []
    row_colors = []
    for name in method_names:
        cells = [rows_data[o]["methods"][name] for o in rows_data]
        guarantee = first[name]["guarantee"]
        mean_gain = sum(c["exploitation_gain_mean"] for c in cells) / len(cells)
        worst_viol = max(c["safety_violation_max"] for c in cells)
        mean_dist = sum(c["distance_from_scbr_mean"] for c in cells) / len(cells)
        held = all(c["guarantee_held"] for c in cells)
        table_rows.append(
            [
                name,
                guarantee,
                f"{mean_gain:+.3f}",
                f"{worst_viol:.2f}" if worst_viol > 1e-8 else "0 (safe)",
                f"{mean_dist:+.3f}",
                "n/a" if guarantee == "none" else ("yes" if held else "NO"),
            ]
        )
        row_colors.append(_GUARANTEE_COLOR.get(guarantee, "#ffffff"))

    _render_table(ax, col_labels, table_rows, row_colors)
    ax.set_title(
        f"Baselines vs ours ({comparison['game']}, {len(rows_data)} opponents, "
        f"{len(comparison['seeds'])} seeds)",
        fontsize=10,
    )
    return ax


def render_baseline_comparison_figure(
    comparison: dict[str, Any], out_dir: str | Path = "results"
) -> Path:
    """Render baseline comparison figure for the plots workflow."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    plot_baseline_comparison(comparison, ax=ax)
    fig.tight_layout()
    path = out / f"baseline_comparison_{comparison['game']}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_showdown_comparison(comparison: dict[str, Any], ax=None):
    """Plot showdown comparison for the plots workflow."""
    ax = ax or plt.subplots(figsize=(11, 3.5))[1]
    rows_data = comparison["opponents"]
    methods = comparison["method_names"]
    col_labels = ["opponent", *methods, "guard \u0394"]
    table_rows = []
    row_colors = []
    for opp, row in rows_data.items():
        m = row["methods"]
        gains = [m[name]["exploitation_gain_mean"] for name in methods]
        guard_delta = (
            m["confidence_guarded"]["exploitation_gain_mean"]
            - m["point_response"]["exploitation_gain_mean"]
        )
        table_rows.append([opp, *[f"{g:+.2f}" for g in gains], f"{guard_delta:+.2f}"])
        row_colors.append("#c8e6c9" if guard_delta > 0.1 else "#ffffff")

    _render_table(ax, col_labels, table_rows, row_colors)
    ax.set_title(
        f"Showdown-only {comparison['game']} — exploitation gain "
        f"({len(rows_data)} opponents, {len(comparison['seeds'])} seeds, "
        f"kappa={comparison['kappa']})",
        fontsize=10,
    )
    return ax


def plot_showdown_safety(comparison: dict[str, Any], ax=None):
    """Plot showdown safety for the plots workflow."""
    ax = ax or plt.subplots(figsize=(11, 3.5))[1]
    rows_data = comparison["opponents"]
    methods = comparison["method_names"]
    v_ref = next(iter(rows_data.values()))["game_value"]
    certified_floor = v_ref - comparison.get("rho_cap", 0.0)
    col_labels = ["opponent", *methods]
    table_rows = []
    safety_values = []
    for opp, row in rows_data.items():
        m = row["methods"]
        vals = [m[name]["min_safety_value"] for name in methods]
        safety_values.append(vals)
        table_rows.append([opp, *[f"{v:+.2f}" for v in vals]])

    _render_table(ax, col_labels, table_rows, ["#ffffff"] * len(table_rows))

    for table in list(ax.tables):
        for (r, c), cell in table.get_celld().items():
            if r == 0 or c == 0:
                continue
            if safety_values[r - 1][c - 1] < certified_floor - 0.01:
                cell.set_facecolor("#ffcdd2")
    ax.set_title(
        f"Worst-round safety (game value {v_ref:+.3f}, certified floor "
        f"{certified_floor:+.3f}); red = breaches the certified floor",
        fontsize=10,
    )
    return ax


def render_showdown_comparison_figure(
    comparison: dict[str, Any], out_dir: str | Path = "results"
) -> Path:
    """Render showdown comparison figure for the plots workflow."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, (ax_gain, ax_safety) = plt.subplots(2, 1, figsize=(11, 7))
    plot_showdown_comparison(comparison, ax=ax_gain)
    plot_showdown_safety(comparison, ax=ax_safety)
    fig.tight_layout()
    path = out / f"showdown_comparison_{comparison['game']}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _showdown_sweep_series(comparison: dict[str, Any], field: str) -> tuple[list, dict]:
    """Compute showdown sweep series for the plots workflow."""
    sweep = comparison["sweep"]
    key_to_val = sweep["key_to_value"]
    rows = comparison["opponents"]
    order = sorted(rows, key=lambda k: key_to_val[k])
    xs = [key_to_val[k] for k in order]
    series = {
        m: [rows[k]["methods"][m][field] for k in order]
        for m in comparison["method_names"]
    }
    series["_scbr_gain"] = [rows[k]["scbr_gain"] for k in order]
    return xs, series


def plot_showdown_sweep(comparison: dict[str, Any], ax_gain=None, ax_safety=None):
    """Plot showdown sweep for the plots workflow."""
    if ax_gain is None or ax_safety is None:
        _, (ax_gain, ax_safety) = plt.subplots(1, 2, figsize=(11, 4))
    xs, gain = _showdown_sweep_series(comparison, "exploitation_gain_mean")
    _, gstd = _showdown_sweep_series(comparison, "exploitation_gain_std")
    _, safe = _showdown_sweep_series(comparison, "min_safety_value")
    param = comparison["sweep"]["param"]
    v_ref = next(iter(comparison["opponents"].values()))["game_value"]
    rho_cap = comparison.get("rho_cap", 0.0)

    ax_gain.plot(
        xs, gain["_scbr_gain"], "k--", alpha=0.5, label="hard-safe ceiling (SCBR)"
    )
    ax_gain.plot(
        xs, gain["passive_public"], "o-", color="#777777", label="passive_public"
    )
    for name, color, marker in (
        ("public_robust", "#1f77b4", "D"),
        ("point_response", "#d62728", "s"),
        ("confidence_guarded", "#2ca02c", "^"),
    ):
        mu, sd = gain[name], gstd[name]
        ax_gain.fill_between(
            xs,
            [a - b for a, b in zip(mu, sd, strict=False)],
            [a + b for a, b in zip(mu, sd, strict=False)],
            color=color,
            alpha=0.15,
        )
        ax_gain.plot(xs, mu, marker + "-", color=color, label=name)
    ax_gain.axhline(0.0, color="k", lw=0.5)
    ax_gain.set_xlabel(f"{param} strength")
    ax_gain.set_ylabel("exploitation gain over v_ref")
    ax_gain.set_title(
        f"Activation curve: {comparison['game']} "
        f"(kappa={comparison.get('kappa')}, {len(comparison['seeds'])} seeds)",
        fontsize=10,
    )
    ax_gain.legend(fontsize=8, loc="upper left")
    ax_gain.grid(alpha=0.3)

    certified_floor = v_ref - rho_cap
    ax_safety.axhline(
        certified_floor,
        color="r",
        ls=":",
        label=f"certified floor {certified_floor:+.3f}",
    )
    ax_safety.axhline(
        v_ref, color="k", ls="--", alpha=0.5, label=f"game value {v_ref:+.3f}"
    )
    ax_safety.plot(
        xs, safe["public_robust"], "D-", color="#1f77b4", label="public_robust"
    )
    ax_safety.plot(
        xs, safe["point_response"], "s-", color="#d62728", label="point_response"
    )
    ax_safety.plot(
        xs,
        safe["confidence_guarded"],
        "^-",
        color="#2ca02c",
        label="confidence_guarded",
    )
    ax_safety.set_xlabel(f"{param} strength")
    ax_safety.set_ylabel("worst-round safety")
    ax_safety.set_title("Certified-floor adherence", fontsize=10)
    ax_safety.legend(fontsize=8, loc="lower left")
    ax_safety.grid(alpha=0.3)
    return ax_gain, ax_safety


def render_showdown_sweep_figure(
    comparison: dict[str, Any], out_dir: str | Path = "results"
) -> Path:
    """Render showdown sweep figure for the plots workflow."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, (ax_gain, ax_safety) = plt.subplots(1, 2, figsize=(11, 4))
    plot_showdown_sweep(comparison, ax_gain=ax_gain, ax_safety=ax_safety)
    fig.tight_layout()
    path = out / f"showdown_sweep_{comparison['game']}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_adversarial_stress(stress: dict[str, Any], ax=None):
    """Plot adversarial stress for the plots workflow."""
    ax = ax or plt.subplots(figsize=(9, 3))[1]
    methods = stress["methods"]
    floor = stress["safety_floor"]

    col_labels = [
        "method",
        "guarantee",
        "realized (mean)",
        "realized (worst)",
        "loss vs floor",
        "held",
    ]
    table_rows = []
    row_colors = []
    for name in stress["method_names"]:
        d = methods[name]
        table_rows.append(
            [
                name,
                d["guarantee"],
                f"{d['realized_value_mean']:+.3f}",
                f"{d['realized_value_worst']:+.3f}",
                f"{d['realized_loss_below_floor']:.2f}"
                if d["realized_loss_below_floor"] > 1e-8
                else "0",
                "n/a"
                if d["guarantee"] == "none"
                else ("yes" if d["guarantee_held"] else "NO"),
            ]
        )
        row_colors.append(_GUARANTEE_COLOR.get(d["guarantee"], "#ffffff"))

    _render_table(ax, col_labels, table_rows, row_colors)
    ax.set_title(
        f"Adversarial stress: realized value vs best-response adversary "
        f"(floor {floor:+.3f})",
        fontsize=10,
    )
    return ax


def render_adversarial_stress_figure(
    stress: dict[str, Any], out_dir: str | Path = "results"
) -> Path:
    """Render adversarial stress figure for the plots workflow."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3))
    plot_adversarial_stress(stress, ax=ax)
    fig.tight_layout()
    path = out / f"adversarial_stress_{stress['game']}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_nonstationary_stress(stress: dict[str, Any], ax=None):
    """Plot nonstationary stress for the plots workflow."""
    ax = ax or plt.subplots(figsize=(10, 3))[1]
    methods = stress["methods"]
    floor = stress["safety_floor"]

    col_labels = [
        "method",
        "guarantee",
        "walk-away (cum.)",
        "peak banked",
        "realized (worst)",
        "held",
    ]
    table_rows = []
    row_colors = []
    for name in stress["method_names"]:
        d = methods[name]
        table_rows.append(
            [
                name,
                d["guarantee"],
                f"{d['cumulative_above_floor']:+.2f}",
                f"{d['peak_above_floor']:+.2f}",
                f"{d['realized_value_worst']:+.3f}",
                "n/a"
                if d["guarantee"] == "none"
                else ("yes" if d["guarantee_held"] else "NO"),
            ]
        )
        row_colors.append(_GUARANTEE_COLOR.get(d["guarantee"], "#ffffff"))

    _render_table(ax, col_labels, table_rows, row_colors)
    kind = stress.get("kind", "nonstationary").replace("_", "-")
    ax.set_title(
        f"Non-stationary stress ({kind}): cumulative walk-away above the floor "
        f"({floor:+.3f})",
        fontsize=10,
    )
    return ax


def render_nonstationary_stress_figure(
    stress: dict[str, Any], out_dir: str | Path = "results"
) -> Path:
    """Render nonstationary stress figure for the plots workflow."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 3))
    plot_nonstationary_stress(stress, ax=ax)
    fig.tight_layout()
    path = out / f"nonstationary_{stress.get('kind', 'stress')}_{stress['game']}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


__all__ = [
    "plot_value_over_rounds",
    "plot_safety_over_rounds",
    "plot_ci_shrinkage",
    "plot_robust_vs_empirical_br",
    "plot_passive_vs_probing",
    "plot_budget_frontier",
    "plot_importance_comparison",
    "plot_coverage",
    "plot_finite_sample_gap",
    "plot_baseline_comparison",
    "plot_adversarial_stress",
    "plot_nonstationary_stress",
    "plot_ablation",
    "render_online_figures",
    "render_ablation_figure",
    "render_probing_figure",
    "render_budget_frontier_figure",
    "render_importance_figure",
    "render_coverage_figure",
    "render_finite_sample_gap_figure",
    "render_baseline_comparison_figure",
    "render_adversarial_stress_figure",
    "render_nonstationary_stress_figure",
]
