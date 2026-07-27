"""Config primitives for safe-observation experiments. See Safe Active De-censoring, Experiments, and supplementary Game Instances and Experimental Setup."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from typing import Any

from ..opponents import opponent_from_spec
from .blueprint import run_kuhn_blueprint
from .online import (
    DEFAULT_SEEDS,
    run_ablation,
    run_adversarial_stress,
    run_baseline_comparison,
    run_budget_frontier,
    run_coverage_experiment,
    run_finite_sample_gap,
    run_importance_comparison,
    run_nonstationary_stress,
    run_online_adaptation,
    run_online_adaptation_replicated,
    run_probing_comparison,
    run_probing_suite,
    run_showdown_comparison,
)


@dataclass
class ConfigRun:
    """Represent config run for the config workflow."""

    kind: str
    name: str
    results: dict[str, Any]
    figures: dict[str, Path] = field(default_factory=dict)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load config for the config workflow."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def _render_online(results: dict[str, Any], out_dir: str | Path) -> dict[str, Path]:
    """Render online for the config workflow."""
    try:
        from ..evaluation.plots import render_online_figures
    except ModuleNotFoundError:
        print("  (matplotlib not installed; skipping figures -- install the viz extra)")
        return {}
    return render_online_figures(results, out_dir=Path(out_dir) / "figures")


def _render_ablation(
    ablation: dict[str, Any], opponent: str, out_dir: str | Path
) -> dict[str, Path]:
    """Render ablation for the config workflow."""
    try:
        from ..evaluation.plots import render_ablation_figure
    except ModuleNotFoundError:
        print("  (matplotlib not installed; skipping figures -- install the viz extra)")
        return {}
    return {
        "ablation": render_ablation_figure(
            ablation, opponent, out_dir=Path(out_dir) / "figures"
        )
    }


def _render_probing(comparison: dict[str, Any], out_dir: str | Path) -> dict[str, Path]:
    """Render probing for the config workflow."""
    try:
        from ..evaluation.plots import render_probing_figure
    except ModuleNotFoundError:
        print("  (matplotlib not installed; skipping figures -- install the viz extra)")
        return {}
    return {
        "probing": render_probing_figure(comparison, out_dir=Path(out_dir) / "figures")
    }


def _render_budget_frontier(
    frontier: dict[str, Any], out_dir: str | Path
) -> dict[str, Path]:
    """Render budget frontier for the config workflow."""
    try:
        from ..evaluation.plots import render_budget_frontier_figure
    except ModuleNotFoundError:
        print("  (matplotlib not installed; skipping figures -- install the viz extra)")
        return {}
    return {
        "budget_frontier": render_budget_frontier_figure(
            frontier, out_dir=Path(out_dir) / "figures"
        )
    }


def _render_importance(
    comparison: dict[str, Any], out_dir: str | Path
) -> dict[str, Path]:
    """Render importance for the config workflow."""
    try:
        from ..evaluation.plots import render_importance_figure
    except ModuleNotFoundError:
        print("  (matplotlib not installed; skipping figures -- install the viz extra)")
        return {}
    return {
        "importance": render_importance_figure(
            comparison, out_dir=Path(out_dir) / "figures"
        )
    }


def _render_coverage(coverage: dict[str, Any], out_dir: str | Path) -> dict[str, Path]:
    """Render coverage for the config workflow."""
    try:
        from ..evaluation.plots import render_coverage_figure
    except ModuleNotFoundError:
        print("  (matplotlib not installed; skipping figures -- install the viz extra)")
        return {}
    return {
        "coverage": render_coverage_figure(coverage, out_dir=Path(out_dir) / "figures")
    }


def _render_finite_gap(gap: dict[str, Any], out_dir: str | Path) -> dict[str, Path]:
    """Render finite gap for the config workflow."""
    try:
        from ..evaluation.plots import render_finite_sample_gap_figure
    except ModuleNotFoundError:
        print("  (matplotlib not installed; skipping figures -- install the viz extra)")
        return {}
    return {
        "finite_gap": render_finite_sample_gap_figure(
            gap, out_dir=Path(out_dir) / "figures"
        )
    }


def _render_baseline(
    comparison: dict[str, Any], out_dir: str | Path
) -> dict[str, Path]:
    """Render baseline for the config workflow."""
    try:
        from ..evaluation.plots import render_baseline_comparison_figure
    except ModuleNotFoundError:
        print("  (matplotlib not installed; skipping figures -- install the viz extra)")
        return {}
    return {
        "baseline_comparison": render_baseline_comparison_figure(
            comparison, out_dir=Path(out_dir) / "figures"
        )
    }


def _render_showdown(
    comparison: dict[str, Any], out_dir: str | Path
) -> dict[str, Path]:
    """Render showdown for the config workflow."""
    try:
        from ..evaluation.plots import render_showdown_comparison_figure
    except ModuleNotFoundError:
        print("  (matplotlib not installed; skipping figures -- install the viz extra)")
        return {}
    return {
        "showdown_comparison": render_showdown_comparison_figure(
            comparison, out_dir=Path(out_dir) / "figures"
        )
    }


def _render_showdown_sweep(
    comparison: dict[str, Any], out_dir: str | Path
) -> dict[str, Path]:
    """Render showdown sweep for the config workflow."""
    try:
        from ..evaluation.plots import render_showdown_sweep_figure
    except ModuleNotFoundError:
        print("  (matplotlib not installed; skipping figures -- install the viz extra)")
        return {}
    return {
        "showdown_sweep": render_showdown_sweep_figure(
            comparison, out_dir=Path(out_dir) / "figures"
        )
    }


def _render_adversarial(stress: dict[str, Any], out_dir: str | Path) -> dict[str, Path]:
    """Render adversarial for the config workflow."""
    try:
        from ..evaluation.plots import render_adversarial_stress_figure
    except ModuleNotFoundError:
        print("  (matplotlib not installed; skipping figures -- install the viz extra)")
        return {}
    return {
        "adversarial_stress": render_adversarial_stress_figure(
            stress, out_dir=Path(out_dir) / "figures"
        )
    }


def _render_nonstationary(
    stress: dict[str, Any], out_dir: str | Path
) -> dict[str, Path]:
    """Render nonstationary for the config workflow."""
    try:
        from ..evaluation.plots import render_nonstationary_stress_figure
    except ModuleNotFoundError:
        print("  (matplotlib not installed; skipping figures -- install the viz extra)")
        return {}
    return {
        "nonstationary_stress": render_nonstationary_stress_figure(
            stress, out_dir=Path(out_dir) / "figures"
        )
    }


def run_config(
    config: str | Path | Mapping[str, Any], figures: bool | None = None
) -> ConfigRun:
    """Run the config experiment for the config workflow."""
    cfg = dict(load_config(config) if isinstance(config, str | Path) else config)
    exp = cfg.get("experiment", {})
    kind = exp.get("kind", "online")
    name = exp.get("name", kind)

    out = cfg.get("output", {})
    out_dir = out.get("dir", "results")
    want_figs = out.get("figures", False) if figures is None else figures

    run = cfg.get("run", {})
    rounds = run.get("rounds", 200)
    episodes = run.get("episodes_per_round", 50)
    seeds = run.get("seeds", list(DEFAULT_SEEDS))
    workers = run.get("workers")
    conf = cfg.get("confidence", {})
    delta = conf.get("delta", 0.05)
    method = conf.get("method", "hoeffding")
    monitoring = conf.get("monitoring", "full")
    eps_safe = cfg.get("safety", {}).get("eps_safe", 0.0)

    if kind == "blueprint":
        solver = cfg.get("solver", {})
        results = run_kuhn_blueprint(
            method=solver.get("method", "lp"),
            iterations=solver.get("iterations", 100_000),
            out_dir=out_dir,
        )
        return ConfigRun(kind=kind, name=name, results=results)

    if kind not in (
        "online",
        "online_replicated",
        "ablation",
        "probing",
        "budget_frontier",
        "importance_comparison",
        "coverage",
        "finite_sample_gap",
        "probing_suite",
        "baseline_comparison",
        "showdown_comparison",
        "showdown_sweep",
        "adversarial_stress",
        "nonstationary_stress",
    ):
        raise ValueError(
            f"unknown experiment kind {kind!r}; expected blueprint, online, "
            f"online_replicated, ablation, probing, budget_frontier, "
            f"importance_comparison, coverage, finite_sample_gap, probing_suite, "
            f"baseline_comparison, or adversarial_stress"
        )

    if kind == "probing_suite":
        probe = cfg.get("probe", {})
        results = run_probing_suite(
            rounds=rounds,
            episodes_per_round=episodes,
            beta=probe.get("beta", 2.0),
            per_round_budget=probe.get("per_round_budget", 0.5),
            delta=delta,
            eps_safe=eps_safe,
            method=method,
            seeds=seeds,
            out_dir=out_dir,
        )
        return ConfigRun(kind=kind, name=name, results=results)

    if kind == "baseline_comparison":
        probe = cfg.get("probe", {})
        suite_game = cfg.get("suite", {}).get("game", "leduc")
        if suite_game == "goofspiel":
            from ..opponents import goofspiel_opponent_suite

            opponents = goofspiel_opponent_suite()
        elif suite_game == "leduc":
            opponents = None
        else:
            raise ValueError(
                f"unknown suite game {suite_game!r}; expected 'leduc' or 'goofspiel'"
            )
        results = run_baseline_comparison(
            opponents,
            rounds=rounds,
            episodes_per_round=episodes,
            delta=delta,
            eps_safe=eps_safe,
            method=method,
            rnr_ps=probe.get("rnr_ps", (0.5,)),
            beta=probe.get("beta", 2.0),
            probe_per_round=probe.get("per_round_budget", 0.5),
            seeds=seeds,
            out_dir=out_dir,
            workers=workers,
        )
        figs = _render_baseline(results, out_dir) if want_figs else {}
        return ConfigRun(kind=kind, name=name, results=results, figures=figs)

    if kind == "showdown_comparison":
        showdown = cfg.get("showdown", {})
        suite_game = cfg.get("suite", {}).get("game", "leduc")
        if suite_game == "leduc":
            opponents = None
        elif suite_game == "holdem":
            from ..opponents import holdem_showdown_opponent_suite

            opponents = holdem_showdown_opponent_suite()
        else:
            raise ValueError(
                f"unknown suite game {suite_game!r}; expected 'leduc' or 'holdem'"
            )
        results = run_showdown_comparison(
            opponents,
            rounds=rounds,
            episodes_per_round=episodes,
            delta=delta,
            eps_safe=eps_safe,
            method=method,
            rho_cap=showdown.get("rho_cap", 0.5),
            kappa=showdown.get("kappa", 0.2),
            safety_debt_max=showdown.get("debt_max", 3.0),
            rnr_p=showdown.get("rnr_p", 0.5),
            seeds=seeds,
            out_dir=out_dir,
            workers=workers,
            latch_threshold=showdown.get("latch_threshold", None),
        )
        figs = _render_showdown(results, out_dir) if want_figs else {}
        return ConfigRun(kind=kind, name=name, results=results, figures=figs)

    if kind == "showdown_sweep":
        from ..opponents import holdem_censored_fold_opponent

        showdown = cfg.get("showdown", {})
        sweep = cfg.get("sweep", {})
        opp_type = sweep.get("opponent", "censored_fold")
        param = sweep.get("param", "leak")
        values = sweep.get("values", [0.0, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.85])
        if opp_type != "censored_fold" or param != "leak":
            raise ValueError(
                "showdown_sweep currently supports opponent='censored_fold', "
                f"param='leak'; got opponent={opp_type!r}, param={param!r}"
            )
        key_to_value = {f"leak{int(v * 100):02d}": float(v) for v in values}
        suite = {
            key: holdem_censored_fold_opponent(leak=val)
            for key, val in key_to_value.items()
        }
        results = run_showdown_comparison(
            suite,
            rounds=rounds,
            episodes_per_round=episodes,
            delta=delta,
            eps_safe=eps_safe,
            method=method,
            rho_cap=showdown.get("rho_cap", 0.5),
            kappa=showdown.get("kappa", 0.2),
            safety_debt_max=showdown.get("debt_max", 3.0),
            rnr_p=showdown.get("rnr_p", 0.5),
            seeds=seeds,
            out_dir=None,
            workers=workers,
        )
        results["sweep"] = {
            "param": param,
            "opponent": opp_type,
            "values": [float(v) for v in values],
            "key_to_value": key_to_value,
        }
        if out_dir is not None:
            from .. import evaluation

            evaluation.save_results(
                results, Path(out_dir) / f"showdown_sweep_{results['game']}.json"
            )
        figs = _render_showdown_sweep(results, out_dir) if want_figs else {}
        return ConfigRun(kind=kind, name=name, results=results, figures=figs)

    if kind == "adversarial_stress":
        probe = cfg.get("probe", {})
        results = run_adversarial_stress(
            rounds=rounds,
            episodes_per_round=episodes,
            delta=delta,
            eps_safe=eps_safe,
            method=method,
            rnr_ps=probe.get("rnr_ps", (0.5,)),
            beta=probe.get("beta", 2.0),
            probe_per_round=probe.get("per_round_budget", 0.5),
            seeds=seeds,
            out_dir=out_dir,
        )
        figs = _render_adversarial(results, out_dir) if want_figs else {}
        return ConfigRun(kind=kind, name=name, results=results, figures=figs)

    if kind == "nonstationary_stress":
        probe = cfg.get("probe", {})
        stress = cfg.get("stress", {})
        results = run_nonstationary_stress(
            kind=stress.get("kind", "lure_then_strike"),
            lure=stress.get("lure", "static_biased"),
            strike_round=stress.get("strike_round"),
            drift_power=stress.get("drift_power", 1.0),
            rounds=rounds,
            episodes_per_round=episodes,
            delta=delta,
            eps_safe=eps_safe,
            method=method,
            rnr_ps=probe.get("rnr_ps", (0.5,)),
            beta=probe.get("beta", 2.0),
            probe_per_round=probe.get("per_round_budget", 0.5),
            seeds=seeds,
            out_dir=out_dir,
        )
        figs = _render_nonstationary(results, out_dir) if want_figs else {}
        return ConfigRun(kind=kind, name=name, results=results, figures=figs)

    opponent = opponent_from_spec(cfg["opponent"])

    if kind == "online":
        results = run_online_adaptation(
            opponent,
            rounds=rounds,
            episodes_per_round=episodes,
            delta=delta,
            eps_safe=eps_safe,
            method=method,
            monitoring=monitoring,
            seed=seeds[0],
            out_dir=out_dir,
        )
        figs = _render_online(results, out_dir) if want_figs else {}
        return ConfigRun(kind=kind, name=name, results=results, figures=figs)

    if kind == "online_replicated":
        results = run_online_adaptation_replicated(
            opponent,
            rounds=rounds,
            episodes_per_round=episodes,
            delta=delta,
            eps_safe=eps_safe,
            method=method,
            monitoring=monitoring,
            seeds=seeds,
            out_dir=out_dir,
        )
        figs = _render_online(results, out_dir) if want_figs else {}
        return ConfigRun(kind=kind, name=name, results=results, figures=figs)

    if kind == "probing":
        probe = cfg.get("probe", {})
        arms = probe.get("arms")
        results = run_probing_comparison(
            opponent,
            arms=arms,
            rounds=rounds,
            episodes_per_round=episodes,
            delta=delta,
            eps_safe=eps_safe,
            method=method,
            seeds=seeds,
            out_dir=out_dir,
        )
        figs = _render_probing(results, out_dir) if want_figs else {}
        return ConfigRun(kind=kind, name=name, results=results, figures=figs)

    if kind == "budget_frontier":
        probe = cfg.get("probe", {})
        results = run_budget_frontier(
            opponent,
            budgets=probe.get("budgets", (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)),
            beta=probe.get("beta", 2.0),
            rounds=rounds,
            episodes_per_round=episodes,
            delta=delta,
            eps_safe=eps_safe,
            method=method,
            seeds=seeds,
            out_dir=out_dir,
        )
        figs = _render_budget_frontier(results, out_dir) if want_figs else {}
        return ConfigRun(kind=kind, name=name, results=results, figures=figs)

    if kind == "importance_comparison":
        probe = cfg.get("probe", {})
        results = run_importance_comparison(
            opponent,
            budgets=probe.get("budgets", (0.05, 0.1, 0.25, 0.5)),
            beta=probe.get("beta", 2.0),
            rounds=rounds,
            episodes_per_round=episodes,
            delta=delta,
            eps_safe=eps_safe,
            method=method,
            seeds=seeds,
            out_dir=out_dir,
        )
        figs = _render_importance(results, out_dir) if want_figs else {}
        return ConfigRun(kind=kind, name=name, results=results, figures=figs)

    if kind == "coverage":
        cov = cfg.get("coverage", {})
        results = run_coverage_experiment(
            opponent,
            deltas=cov.get("deltas", (0.05, 0.1, 0.2)),
            rounds=rounds,
            episodes_per_round=episodes,
            method=method,
            seeds=seeds,
            out_dir=out_dir,
        )
        figs = _render_coverage(results, out_dir) if want_figs else {}
        return ConfigRun(kind=kind, name=name, results=results, figures=figs)

    if kind == "finite_sample_gap":
        results = run_finite_sample_gap(
            opponent,
            rounds=rounds,
            episodes_per_round=episodes,
            delta=delta,
            eps_safe=eps_safe,
            method=method,
            seeds=seeds,
            out_dir=out_dir,
        )
        figs = _render_finite_gap(results, out_dir) if want_figs else {}
        return ConfigRun(kind=kind, name=name, results=results, figures=figs)

    abl = cfg.get("ablation", {})
    ablation = run_ablation(
        opponent,
        deltas=abl.get("deltas", (0.01, 0.05, 0.1, 0.2)),
        eps_safes=abl.get("eps_safes", (0.0, 0.05, 0.1, 0.2)),
        methods=abl.get("methods", ("hoeffding", "empirical_bernstein")),
        rounds=rounds,
        episodes_per_round=episodes,
        seeds=seeds,
        out_dir=out_dir,
    )
    figs = _render_ablation(ablation, opponent.name, out_dir) if want_figs else {}
    return ConfigRun(kind=kind, name=name, results=ablation, figures=figs)


def summarize(run: ConfigRun) -> str:
    """Summarize the supplied experiment records."""
    r = run.results
    if run.kind == "blueprint":
        return (
            f"[{run.name}] blueprint value {r['value_player1']:.6f} "
            f"(abs error {r['abs_error']:.2e})"
        )
    if run.kind == "ablation":
        axes = ", ".join(f"{axis} ({len(cells)} cells)" for axis, cells in r.items())
        return f"[{run.name}] ablation over {axes}"
    if run.kind == "probing":
        arms = r["arms"]
        parts = ", ".join(
            f"{name} {a['exploitation_gain_mean']:+.3f}"
            f"(spent {a['budget_spent_mean']:.1f})"
            for name, a in arms.items()
        )
        return f"[{run.name}] vs {r['opponent']} probing: {parts}"
    if run.kind == "budget_frontier":
        cells = r["cells"]
        span = (
            f"{cells[0]['exploitation_gain_mean']:+.3f} -> "
            f"{cells[-1]['exploitation_gain_mean']:+.3f}"
        )
        return (
            f"[{run.name}] vs {r['opponent']} budget frontier "
            f"(SCBR {r['scbr_gain']:+.3f}): gain {span} over budgets {r['budgets']}"
        )
    if run.kind == "importance_comparison":
        advs = ", ".join(
            f"rho={c['per_round_budget']} adv {c['sensitivity_advantage']:+.3f}"
            for c in r["cells"]
        )
        return f"[{run.name}] vs {r['opponent']} importance (sensitivity adv): {advs}"
    if run.kind == "coverage":
        worst = min(c["arms"]["time_uniform"]["anytime_coverage"] for c in r["cells"])
        nu = min(c["arms"]["no_union"]["anytime_coverage"] for c in r["cells"])
        return (
            f"[{run.name}] vs {r['opponent']} coverage (Thm 2): time-uniform anytime "
            f">= {worst:.3f}, no-union as low as {nu:.3f} (union bound necessary)"
        )
    if run.kind == "finite_sample_gap":
        return (
            f"[{run.name}] vs {r['opponent']} finite-sample gap (Thm 4): "
            f"{r['early_gap_mean']:+.4f} -> {r['late_gap_mean']:+.4f}, "
            f"gap <= {r['bound_constant']:.2f} * radius"
        )
    if run.kind == "probing_suite":
        helped = [name for name, row in r["opponents"].items() if row["probing_helps"]]
        return (
            f"[{run.name}] probing suite over {len(r['opponents'])} opponents; "
            f"budget helps: {', '.join(helped) or 'none'}"
        )
    if run.kind == "baseline_comparison":
        names = r["method_names"]
        held = [
            name
            for name in names
            if all(
                o["methods"][name]["guarantee_held"] for o in r["opponents"].values()
            )
        ]
        return (
            f"[{run.name}] {len(names)} methods x {len(r['opponents'])} opponents; "
            f"guarantee held by: {', '.join(held)}"
        )
    if run.kind == "showdown_comparison":
        deltas = {
            opp: (
                row["methods"]["confidence_guarded"]["exploitation_gain_mean"]
                - row["methods"]["point_response"]["exploitation_gain_mean"]
            )
            for opp, row in r["opponents"].items()
        }
        best = max(deltas.items(), key=lambda kv: kv[1])
        helped = [opp for opp, d in deltas.items() if d > 0.1]
        return (
            f"[{run.name}] showdown-only ({len(r['opponents'])} opponents, "
            f"kappa={r['kappa']}); guard recovers exploitation on: "
            f"{', '.join(helped) or 'none'} (best {best[0]} {best[1]:+.2f})"
        )
    if run.kind == "showdown_sweep":
        sweep = r["sweep"]
        k2v = sweep["key_to_value"]
        order = sorted(r["opponents"], key=lambda k: k2v[k])
        deltas = [
            (
                k2v[k],
                r["opponents"][k]["methods"]["confidence_guarded"][
                    "exploitation_gain_mean"
                ]
                - r["opponents"][k]["methods"]["point_response"][
                    "exploitation_gain_mean"
                ],
            )
            for k in order
        ]
        activated = [v for v, d in deltas if d > 0.1]
        knee = f"{min(activated):.2f}" if activated else "none"
        top = deltas[-1]
        return (
            f"[{run.name}] censored-fold {sweep['param']} sweep "
            f"({len(order)} points, kappa={r['kappa']}); guard activates at "
            f"{sweep['param']}>={knee}, max guard delta {top[1]:+.2f} at "
            f"{sweep['param']}={top[0]:.2f}"
        )
    if run.kind == "adversarial_stress":
        methods = r["methods"]
        worst = min((m["realized_value_worst"], name) for name, m in methods.items())
        held = [n for n in r["method_names"] if methods[n]["guarantee_held"]]
        return (
            f"[{run.name}] adversarial stress (floor {r['safety_floor']:+.3f}): "
            f"worst realized {worst[0]:+.3f} ({worst[1]}); "
            f"guarantee held by: {', '.join(held)}"
        )
    if run.kind == "nonstationary_stress":
        methods = r["methods"]
        best = max((m["cumulative_above_floor"], name) for name, m in methods.items())
        worst = min((m["cumulative_above_floor"], name) for name, m in methods.items())
        return (
            f"[{run.name}] {r['kind']} (floor {r['safety_floor']:+.3f}): "
            f"best walk-away {best[0]:+.2f} ({best[1]}), "
            f"worst {worst[0]:+.2f} ({worst[1]}); "
            f"gift_based walk-away {methods['gift_based']['cumulative_above_floor']:+.2f} "
            f"(peak {methods['gift_based']['peak_above_floor']:+.2f})"
        )
    if run.kind == "online_replicated":
        return (
            f"[{run.name}] vs {r['opponent']}: gain "
            f"{r['exploitation_gain_mean']:+.4f} +/- {r['exploitation_gain_std']:.4f}, "
            f"min safety {r['min_safety_value']:.4f}, safe={r['safety_preserved']}"
        )
    return (
        f"[{run.name}] vs {r['opponent']}: gain {r['exploitation_gain']:+.4f}, "
        f"min safety {r['min_safety_value']:.4f}, safe={r['safety_preserved']}"
    )


__all__ = ["ConfigRun", "load_config", "run_config", "summarize"]
