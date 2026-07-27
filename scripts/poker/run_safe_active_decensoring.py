"""Run matched-budget Safe Active De-censoring with held-out reveal data.

Each empirical cell spends a fixed total budget. A blueprint pilot supplies
public confidence rows and freezes the route; an independent reveal batch then
supplies committing-sequence mass rows. Empirical modes never use the benchmark
opponent for routing, response construction, or fallback.
"""

from collections.abc import Mapping, Sequence
from concurrent.futures import TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import dataclass
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import sys
from typing import Any

from pebble import ProcessPool

from safe_observation import native
from safe_observation.confidence import (
    OpponentEvidenceStore,
    OpponentEvidenceStore as _Store,
    allocate_simultaneous_delta,
    empirical_bernstein_interval,
    hoeffding_interval,
)
from safe_observation.opponents import Opponent, holdem_equilibrium_opponent
from safe_observation.payoff import build as build_payoff
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import (
    agent_showdown_reach,
    opponent_reach_weights,
    robust_safe_response_linear,
    robust_safe_response_probe,
    robust_safe_response_public,
    safety_constrained_best_response,
    safety_verifier,
    solve_blueprint,
)
from scripts.poker.estimate_identifiable_value import (
    _fold_action_indices,
)
from scripts.poker.evaluate_public_routing import _build_gate_suite
from scripts.poker.run_active_decensoring import _empirical_event_constraints
from scripts.poker.run_turn_river_methods import _perturb, _top_rank

GAME = sys.argv[1] if len(sys.argv) > 1 else "holdem_tr_b2"
RHO = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
N_GRID = [
    int(x)
    for x in (
        sys.argv[3].split(",") if len(sys.argv) > 3 else ["1000", "10000", "100000"]
    )
]
SEEDS = (
    [2026 + i for i in range(int(sys.argv[4]))]
    if len(sys.argv) > 4
    else [2026 + i for i in range(10)]
)
NPROC = int(sys.argv[5]) if len(sys.argv) > 5 else 10


CELL_TIMEOUT = (
    float(sys.argv[6])
    if len(sys.argv) > 6
    else float(os.environ.get("SAD_CELL_TIMEOUT", "600"))
)
DELTA = 0.1
METHOD = "empirical_bernstein"
BETA = 1e6
TOL = 1e-6
PUBLIC_PILOT_FRACTION = float(os.environ.get("SAD_PUBLIC_FRACTION", "0.2"))
ROUTE_SCREEN_DELTA = float(os.environ.get("SAD_ROUTE_SCREEN_DELTA", "0.1"))
PIPELINE_VERSION = "sample_split_v3_screened"
EXPERIMENT_CONFIG = {
    "game": GAME,
    "rho": RHO,
    "delta": DELTA,
    "method": METHOD,
    "beta": BETA,
    "public_pilot_fraction": PUBLIC_PILOT_FRACTION,
    "route_screen_delta": ROUTE_SCREEN_DELTA,
    "pipeline_version": PIPELINE_VERSION,
}
CONFIG_FINGERPRINT = json.dumps(
    EXPERIMENT_CONFIG, sort_keys=True, separators=(",", ":")
)
MODES = ["cpub", "passive", "random", "sad", "oracle_target", "oracle"]
LEAKS = [
    "tr_equilibrium",
    "river_overfold_w80",
    "turn_overfold_w70",
    "revealed_call_strong",
]


ABLATION = os.environ.get("SAD_ABLATION") == "1"
if ABLATION:
    MODES = [
        "cpub",
        "kappa",
        "dpub",
        "dpub_kappa",
        "random",
        "oracle_target",
        "oracle",
    ]


if os.environ.get("SAD_CONC") == "1":
    LEAKS = ["tr_equilibrium", "conc_line"]
if requested_modes := os.environ.get("SAD_MODES"):
    MODES = [mode for mode in requested_modes.split(",") if mode]
if requested_leaks := os.environ.get("SAD_LEAKS"):
    LEAKS = [name for name in requested_leaks.split(",") if name]


_W: dict[str, Any] = {}


def experiment_provenance() -> dict[str, Any]:
    """Return the configuration fields carried by every checkpoint row."""
    return {
        **EXPERIMENT_CONFIG,
        "config_fingerprint": CONFIG_FINGERPRINT,
        "public_pilot_fraction_config": PUBLIC_PILOT_FRACTION,
    }


def legacy_metadata_matches_current(metadata: Mapping[str, Any]) -> bool:
    """Check whether an explicitly migrated legacy summary matches this run."""
    try:
        return (
            metadata.get("game") == GAME
            and float(metadata.get("rho")) == RHO
            and float(metadata.get("delta")) == DELTA
            and metadata.get("method") == METHOD
            and float(metadata.get("public_pilot_fraction")) == PUBLIC_PILOT_FRACTION
            and metadata.get("protocol") == PIPELINE_VERSION
        )
    except (TypeError, ValueError):
        return False


def cell_failed(row: Mapping[str, Any]) -> bool:
    """Return whether a checkpoint row lacks a completed floor audit."""
    return bool(
        row.get("cell_failed") or row.get("lp_timeout") or row.get("execution_error")
    )


@dataclass(frozen=True)
class SamplingPlan:
    """A reproducible split between route selection and reveal evidence."""

    total_budget: int
    public_budget: int
    reveal_budget: int
    public_seed: int
    reveal_seed: int

    @property
    def public_fraction(self) -> float:
        """Return the charged fraction reserved for the public pilot."""
        return self.public_budget / self.total_budget if self.total_budget else 0.0


def plan_sampling_batches(
    total_budget: int,
    seed: int,
    public_fraction: float = PUBLIC_PILOT_FRACTION,
) -> SamplingPlan:
    """Fix disjoint public and reveal batches before observing either sample."""
    if total_budget < 2:
        raise ValueError("total_budget must leave at least one sample per batch")
    if not 0.0 < public_fraction < 1.0:
        raise ValueError("public_fraction must be in the open interval (0, 1)")
    public_budget = max(1, min(total_budget - 1, int(total_budget * public_fraction)))
    return SamplingPlan(
        total_budget=total_budget,
        public_budget=public_budget,
        reveal_budget=total_budget - public_budget,
        public_seed=2 * seed,
        reveal_seed=2 * seed + 1,
    )


def matched_capture(
    row: Mapping[str, Any],
    public_by_cell: Mapping[tuple[str, int, int], Mapping[str, Any]],
    oracle_by_opponent: Mapping[str, Mapping[str, Any]],
) -> float:
    """Measure realized lift relative to the matched public arm and oracle."""
    if row["mode"] == "cpub":
        return 0.0
    if row["mode"] == "oracle":
        return 1.0
    public = public_by_cell.get(
        (row["opponent"], int(row["episodes"]), int(row["seed"]))
    )
    if public is None:
        return float("nan")
    lower = float(public["realized"])
    upper = float(oracle_by_opponent[row["opponent"]]["realized"])
    realized = float(row["realized"])
    return (realized - lower) / (upper - lower) if upper - lower > 1e-9 else 0.0


def select_checkpoint_rows(
    records: list[dict[str, Any]],
    tasks: list[tuple[str, str, int, int]],
    *,
    allow_legacy_config: bool = False,
) -> tuple[list[dict[str, Any]], int, int]:
    """Keep the latest successful compatible row for each requested cell."""
    requested = set(tasks)
    latest: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    ignored = 0
    retried = 0
    for row in records:
        try:
            key = (
                str(row["mode"]),
                str(row["opponent"]),
                int(row["episodes"]),
                int(row["seed"]),
            )
        except (KeyError, TypeError, ValueError):
            ignored += 1
            continue
        configured_fraction = row.get(
            "public_pilot_fraction_config", row.get("public_fraction", -1.0)
        )
        try:
            configured_fraction = float(configured_fraction)
        except (TypeError, ValueError):
            ignored += 1
            continue
        fingerprint = row.get("config_fingerprint")
        config_compatible = (
            fingerprint == CONFIG_FINGERPRINT
            if fingerprint is not None
            else allow_legacy_config
        )
        if (
            key not in requested
            or row.get("pipeline_version") != PIPELINE_VERSION
            or configured_fraction != PUBLIC_PILOT_FRACTION
            or not config_compatible
        ):
            ignored += 1
            continue
        if cell_failed(row):
            retried += 1
            continue
        normalized = {**row, **experiment_provenance()}
        total_budget = int(normalized.get("total_budget", key[2]))
        public_budget = normalized.get("public_budget")
        if public_budget is not None:
            normalized["public_fraction"] = (
                float(public_budget) / total_budget if total_budget > 0 else 0.0
            )
        latest[key] = normalized
    return [latest[task] for task in tasks if task in latest], ignored, retried


def _is_river(label: str) -> bool:
    """Return whether a target belongs to the river round."""
    return "/" in (label.split("|", 1)[1] if "|" in label else "")


def _try_solve(fn):
    """Return ``None`` only for a known empty confidence set."""
    try:
        return fn()
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        message = str(exc).lower()
        expected_empty_set = (
            isinstance(exc, ValueError)
            and ("infeasible" in message or "empty" in message)
        ) or (
            type(exc).__name__ == "PanicException"
            and "c_t is empty" in message
            and "intervals are inconsistent" in message
        )
        if expected_empty_set:
            return None
        raise


def _build_probe_behavior(weights: dict[str, float]) -> dict[str, list[float]]:
    """Solve the floor-constrained weighted-reach probe."""
    sf0 = _W["sf0"]
    pr = robust_safe_response_probe(
        _W["triv_iv"],
        _W["cont_beh"],
        weights,
        v_ref=_W["v_ref"],
        eps_safe=RHO,
        beta=BETA,
        rho=0.0,
        game=GAME,
    )
    return sf0.behavior_from_realization(list(pr.realization))


def _public_action_rates(
    groups: Mapping[str, Sequence[str]],
    info_by: Mapping[str, Any],
    realization: Sequence[float],
    agent_reach: Mapping[str, float],
) -> dict[str, list[float]]:
    """Compute exact public action rates for a known reference strategy."""
    rates: dict[str, list[float]] = {}
    for key, members in groups.items():
        if not members:
            continue
        n_actions = len(info_by[members[0]].children)
        numerator = [0.0] * n_actions
        denominator = 0.0
        for label in members:
            info = info_by[label]
            reach = float(agent_reach.get(label, 0.0))
            denominator += reach * realization[info.parent_seq]
            for action, (_name, child) in enumerate(info.children):
                numerator[action] += reach * realization[child]
        if denominator > 1e-12:
            rates[key] = [value / denominator for value in numerator]
    return rates


def _init(
    probes: dict[str, Any] | None = None, kappa: dict[str, float] | None = None
) -> None:
    """Initialize process-local state for parallel experiment workers."""
    sf0 = compile_game(GAME, 0)
    sf1 = compile_game(GAME, 1)
    payoff = build_payoff(GAME)
    bp = solve_blueprint(GAME, method="lp")
    v_ref = bp.value
    omega_bp = opponent_reach_weights(bp.realization, game=GAME)
    fold_idx = _fold_action_indices(sf1)
    groups = OpponentEvidenceStore.for_game(GAME).public_groups()
    info_by = {i.label: i for i in sf1.info_sets}
    eqb = holdem_equilibrium_opponent(GAME).behavior
    y_eq = list(Opponent(name="eq", behavior=eqb, game=GAME).realization())
    bp_behavior = sf0.behavior_from_realization(bp.realization)
    eq_public_rates = _public_action_rates(groups, info_by, y_eq, omega_bp)
    triv_iv = {i.label: [(0.0, 1.0)] * len(i.children) for i in sf1.info_sets}
    cont_beh = {}
    for info in sf1.info_sets:
        n = len(info.children)
        fi = fold_idx.get(info.label)
        if fi is None:
            cont_beh[info.label] = [1.0 / n] * n
        else:
            nf = [k for k in range(n) if k != fi]
            cont_beh[info.label] = [
                (1.0 / len(nf)) if k in nf else 0.0 for k in range(n)
            ]
    actions = {i.label: [a for a, _ in i.children] for i in sf1.info_sets}
    suite = _build_gate_suite(GAME, actions, eqb)

    if os.environ.get("SAD_CONC") == "1":
        cheap = {"cc/7sc", "cc/Ahc"}
        deep = {"cpac/7sc", "cpac/Ahc", "cppac/7sc", "cpp", "cpa"}
        conc = _perturb(
            eqb,
            actions,
            lambda hole, hist, acts: (
                hist in cheap and _top_rank(hole) >= 11 and "c" in acts
            ),
            "c",
            0.6,
        )
        conc = _perturb(
            conc,
            actions,
            lambda hole, hist, acts: (
                hist in deep and _top_rank(hole) >= 11 and "f" in acts
            ),
            "f",
            0.7,
        )
        suite["conc_line"] = conc

    _W.update(
        sf0=sf0,
        sf1=sf1,
        payoff=payoff,
        v_ref=v_ref,
        floor=v_ref - RHO,
        omega_bp=omega_bp,
        fold_idx=fold_idx,
        groups=groups,
        info_by=info_by,
        y_eq=y_eq,
        bp_behavior=bp_behavior,
        bp_realization=list(bp.realization),
        eq_public_rates=eq_public_rates,
        triv_iv=triv_iv,
        cont_beh=cont_beh,
        suite=suite,
        probes=probes or {},
        kappa=kappa or {},
    )


def _public_anomaly_weights(y_ref: list[float]) -> dict[str, float]:
    """Compute the population public anomaly for offline diagnostics."""
    sf1, omega_bp, y_eq = _W["sf1"], _W["omega_bp"], _W["y_eq"]
    by_hist: dict[str, list[Any]] = {}
    for info in sf1.info_sets:
        h = info.label.split("|", 1)[1] if "|" in info.label else ""
        by_hist.setdefault(h, []).append(info)

    anomaly: dict[str, float] = {}
    for h, infos in by_hist.items():
        nA = len(infos[0].children)
        num_ref = [0.0] * nA
        num_eq = [0.0] * nA
        den_ref = den_eq = 0.0
        for info in infos:
            w = omega_bp.get(info.label, 0.0)
            if w <= 0.0:
                continue
            den_ref += w * y_ref[info.parent_seq]
            den_eq += w * y_eq[info.parent_seq]
            for a, (_ac, child) in enumerate(info.children):
                num_ref[a] += w * y_ref[child]
                num_eq[a] += w * y_eq[child]
        if den_ref <= 1e-12 or den_eq <= 1e-12:
            continue
        tv = 0.5 * sum(
            abs(num_ref[a] / den_ref - num_eq[a] / den_eq) for a in range(nA)
        )
        if tv > 1e-6:
            anomaly[h] = tv

    weights: dict[str, float] = {}
    for info in sf1.info_sets:
        h = info.label.split("|", 1)[1] if "|" in info.label else ""
        if "/" not in h:
            continue
        score = sum(tv for hk, tv in anomaly.items() if h.startswith(hk))
        if score > 1e-6:
            weights[info.label] = score
    return weights


def public_anomaly_weights(
    public_counts: Mapping[str, Sequence[int]],
    equilibrium_rates: Mapping[str, Sequence[float]],
    target_labels: Sequence[str],
    screen_delta: float | None = None,
) -> dict[str, float]:
    """Score targets using only aggregated counts from the public pilot."""
    n_coordinates = sum(
        len(counts)
        for key, counts in public_counts.items()
        if key in equilibrium_rates and len(equilibrium_rates[key]) == len(counts)
    )
    per_coordinate_delta = (
        screen_delta / max(1, n_coordinates) if screen_delta is not None else None
    )
    anomaly: dict[str, float] = {}
    for key, counts in public_counts.items():
        reference = equilibrium_rates.get(key)
        visits = sum(counts)
        if reference is None or visits <= 0 or len(reference) != len(counts):
            continue
        tv_hat = 0.5 * sum(
            abs(count / visits - float(ref))
            for count, ref in zip(counts, reference, strict=True)
        )
        if per_coordinate_delta is None:
            tv = tv_hat
        else:
            radius = math.sqrt(
                math.log(2.0 / per_coordinate_delta) / (2.0 * visits)
            )
            tv = max(0.0, tv_hat - 0.5 * len(counts) * radius)
        if tv > 1e-6:
            anomaly[key] = tv

    weights: dict[str, float] = {}
    for label in target_labels:
        history = label.split("|", 1)[1] if "|" in label else ""
        if "/" not in history:
            continue
        score = sum(value for key, value in anomaly.items() if history.startswith(key))
        if score > 1e-6:
            weights[label] = score
    return weights


def empirical_route_weights(
    mode: str,
    public_counts: Mapping[str, Sequence[int]],
    route_seed: int,
) -> dict[str, float]:
    """Build a route without private labels or population opponent inputs."""
    import random

    targets = [info.label for info in _W["sf1"].info_sets if _is_river(info.label)]
    if mode == "random":
        rng = random.Random(route_seed)
        return {label: rng.random() for label in targets}
    if mode == "kappa":
        return _kappa_weights()
    dpub = public_anomaly_weights(
        public_counts,
        _W["eq_public_rates"],
        targets,
        screen_delta=ROUTE_SCREEN_DELTA,
    )
    if mode in ("sad", "dpub"):
        return dpub
    if mode == "dpub_kappa":
        kappa = _W["kappa"]
        return {
            label: score * kappa.get(label, 0.0)
            for label, score in dpub.items()
            if kappa.get(label, 0.0) > 1e-9
        }
    raise ValueError(f"mode {mode!r} is not an empirical routing mode")


def _leak_weights(y_star: list[float]) -> dict[str, float]:
    """Compute weights for leak for the run safe active de-censoring workflow."""
    sf1, fold_idx, y_eq = _W["sf1"], _W["fold_idx"], _W["y_eq"]
    w: dict[str, float] = {}
    for info in sf1.info_sets:
        if not _is_river(info.label):
            continue
        fi = fold_idx.get(info.label)
        dev = sum(
            abs(y_star[child] - y_eq[child])
            for a, (_ac, child) in enumerate(info.children)
            if not (fi is not None and a == fi)
        )
        if dev > 1e-6:
            w[info.label] = dev
    return w


def _kappa_weights() -> dict[str, float]:
    """Compute weights for kappa for the run safe active de-censoring workflow."""
    return {i: k for i, k in _W["kappa"].items() if k > 1e-9}


def _dpub_kappa_weights(y_ref: list[float]) -> dict[str, float]:
    """Compute weights for dpub kappa."""
    d = _public_anomaly_weights(y_ref)
    k = _W["kappa"]
    return {i: d[i] * k.get(i, 0.0) for i in d if k.get(i, 0.0) > 1e-9}


def _kappa_task(label: str) -> tuple[str, float]:
    """Compute kappa task for the run safe active de-censoring workflow."""
    pr = _try_solve(
        lambda: robust_safe_response_probe(
            _W["triv_iv"],
            _W["cont_beh"],
            {label: 1.0},
            v_ref=_W["v_ref"],
            eps_safe=RHO,
            beta=BETA,
            rho=0.0,
            game=GAME,
        )
    )
    if pr is None:
        return label, 0.0
    x = list(pr.realization)
    return label, float(opponent_reach_weights(x, game=GAME).get(label, 0.0))


def _probe_task(task: tuple[str, str]) -> tuple[str, Any, Any]:
    """Build a population-limit probe for offline mechanism diagnostics."""
    import random as _random

    mode, name = task
    behavior = _W["suite"][name]
    y_star = list(Opponent(name=name, behavior=behavior, game=GAME).realization())
    if mode == "random":
        rng = _random.Random(hash(name) & 0xFFFF)
        weights = {
            i.label: rng.random() for i in _W["sf1"].info_sets if _is_river(i.label)
        }
    elif mode in ("sad", "dpub"):
        weights = _public_anomaly_weights(y_star)
    elif mode == "kappa":
        weights = _kappa_weights()
    elif mode == "dpub_kappa":
        weights = _dpub_kappa_weights(y_star)
    elif mode == "oracle_target":
        weights = _leak_weights(y_star)
    else:
        raise ValueError(mode)
    if not weights:
        return f"{mode}|{name}", None, None
    beh = _build_probe_behavior(weights)
    sv = safety_verifier(_W["sf0"].realization_from_behavior(beh), game=GAME).value
    return f"{mode}|{name}", beh, sv


def _record_evidence(
    show: Mapping[str, Sequence[int]],
    fold: Mapping[str, Sequence[int]],
) -> OpponentEvidenceStore:
    """Aggregate showdown and fold observations in one public evidence store."""
    evidence = _Store.for_game(GAME)
    for label, counts in show.items():
        evidence.record(label, counts)
    for label, counts in fold.items():
        evidence.record(label, counts)
    return evidence


def _cell_probe(
    mode: str,
    name: str,
    public_counts: Mapping[str, Sequence[int]],
    route_seed: int,
) -> tuple[dict[str, list[float]], float, str, int]:
    """Freeze a probe after the pilot and before collecting reveal evidence."""
    if mode == "oracle_target":
        behavior = _W["suite"][name]
        y_oracle = list(Opponent(name=name, behavior=behavior, game=GAME).realization())
        weights = _leak_weights(y_oracle)
        source = "oracle_truth"
    else:
        weights = empirical_route_weights(mode, public_counts, route_seed)
        source = (
            "sampled_public_pilot"
            if mode in ("sad", "dpub", "dpub_kappa")
            else "data_independent"
        )
    if not weights:
        return _W["bp_behavior"], _W["v_ref"], f"{source}:blueprint", 0
    probe = _try_solve(lambda: _build_probe_behavior(weights))
    if probe is None:
        return _W["bp_behavior"], _W["v_ref"], f"{source}:blueprint", len(weights)
    probe_realization = _W["sf0"].realization_from_behavior(probe)
    probe_safety = safety_verifier(probe_realization, game=GAME).value
    return probe, probe_safety, source, len(weights)


def _empirical_mass_constraints(sf1, show, sd_reach, fold_idx, episodes, delta, method):
    """Compute empirical mass constraints for the run safe active de-censoring workflow."""

    pins: list[tuple[Any, int, int, float]] = []
    for info in sf1.info_sets:
        row = sd_reach.get(info.label)
        if row is None:
            continue
        fi = fold_idx.get(info.label)
        for a, (_act, child) in enumerate(info.children):
            if fi is not None and a == fi:
                continue
            if a >= len(row):
                continue
            w_sd, committing = row[a]
            if committing and w_sd > 1e-12:
                pins.append((info, a, child, w_sd))
    n_pairs = len(pins)
    per = delta / max(1, n_pairs)

    entries: list[tuple[int, int, float]] = []
    h: list[float] = []
    meta: list[tuple[str, int]] = []
    row_i = 0
    for info, a, child, w_sd in pins:
        cnt = show.get(info.label, [0] * len(info.children))
        mhat = cnt[a] / episodes
        if method == "empirical_bernstein":
            lo, hi = empirical_bernstein_interval(mhat, episodes, per)
        else:
            lo, hi = hoeffding_interval(mhat, episodes, per)
        if hi < w_sd:
            entries.append((row_i, child, w_sd))
            h.append(hi)
            meta.append((info.label, a))
            row_i += 1
        if lo > 0.0:
            entries.append((row_i, child, -w_sd))
            h.append(-lo)
            meta.append((info.label, a))
            row_i += 1
    return entries, h, meta, n_pairs


def _mass_pair_count(sf1, showdown_reach, fold_idx) -> int:
    """Count committing-mass coordinates before allocating failure probability."""
    total = 0
    for info in sf1.info_sets:
        row = showdown_reach.get(info.label)
        if row is None:
            continue
        fold_action = fold_idx.get(info.label)
        for action, (_name, _child) in enumerate(info.children):
            if fold_action is not None and action == fold_action:
                continue
            if action < len(row) and row[action][1] and row[action][0] > 1e-12:
                total += 1
    return total


def _passive_pair_count(sf1, evidence, reach, fold_idx) -> int:
    """Count observed non-fold behavioral coordinates in the passive arm."""
    total = 0
    for info in sf1.info_sets:
        if reach.get(info.label, 0.0) <= 0.0 or evidence.visits(info.label) <= 0:
            continue
        fold_action = fold_idx.get(info.label)
        total += sum(1 for action in range(len(info.children)) if action != fold_action)
    return total


def deploy_cell(mode: str, name: str, total_budget: int, seed: int) -> dict[str, Any]:
    """Run one sample-split acquisition and robust deployment cell."""
    sf1, payoff, v_ref = _W["sf1"], _W["payoff"], _W["v_ref"]
    groups, omega_bp, fold_idx = _W["groups"], _W["omega_bp"], _W["fold_idx"]
    behavior = _W["suite"][name]

    if mode == "oracle":
        y_oracle = list(Opponent(name=name, behavior=behavior, game=GAME).realization())
        response = safety_constrained_best_response(
            behavior, v_ref=v_ref, eps_safe=RHO, game=GAME
        )
        realization = list(response.realization)
        realized = payoff.bilinear(realization, y_oracle) - v_ref
        return {
            **experiment_provenance(),
            "certified": realized,
            "realized": realized,
            "safety": safety_verifier(realization, game=GAME).value,
            "public_budget": 0,
            "reveal_budget": 0,
            "total_budget": 0,
            "public_seed": None,
            "reveal_seed": None,
            "protocol": "population_oracle_v1",
            "pipeline_version": PIPELINE_VERSION,
            "public_pilot_fraction_config": PUBLIC_PILOT_FRACTION,
            "public_fraction": 0.0,
            "population_inputs_used": True,
            "response_source": "oracle_truth",
            "delta_global": None,
            "delta_public": None,
            "delta_reveal": None,
            "public_stat_rows": 0,
            "reveal_stat_rows": 0,
        }

    if mode == "cpub":
        plan = SamplingPlan(
            total_budget=total_budget,
            public_budget=total_budget,
            reveal_budget=0,
            public_seed=2 * seed,
            reveal_seed=2 * seed + 1,
        )
    else:
        plan = plan_sampling_batches(total_budget, seed)

    _pay, pilot_show, pilot_fold = native.simulate_showdown(
        GAME,
        _W["bp_behavior"],
        behavior,
        plan.public_budget,
        plan.public_seed,
    )
    pilot = _record_evidence(pilot_show, pilot_fold)
    public_counts = pilot.public_counts()

    box_infeasible = False
    public_infeasible = False
    n_pairs = 0
    probe_safety = v_ref
    route_targets = 0
    route_source = "none"

    if mode == "cpub":
        delta_budget = allocate_simultaneous_delta(
            DELTA,
            public_rows=pilot.num_public_pairs,
            reveal_rows=0,
        )
        assert delta_budget.public_row_delta is not None
        public_intervals = pilot.public_intervals(
            delta_budget.public_row_delta,
            method=METHOD,
            union_bound=False,
        )
        response = _try_solve(
            lambda: robust_safe_response_public(
                groups,
                public_intervals,
                v_ref=v_ref,
                eps_safe=RHO,
                game=GAME,
                weights=omega_bp,
            )
        )
        response_source = "empirical_cpub"
    else:
        if mode == "passive":
            agent_behavior = _W["bp_behavior"]
            route_source = "blueprint_passive"
        else:
            agent_behavior, probe_safety, route_source, route_targets = _cell_probe(
                mode,
                name,
                public_counts,
                plan.public_seed,
            )

        _pay, reveal_show, _reveal_fold = native.simulate_showdown(
            GAME,
            agent_behavior,
            behavior,
            plan.reveal_budget,
            plan.reveal_seed,
        )
        agent_realization = _W["sf0"].realization_from_behavior(agent_behavior)
        if mode == "passive":
            point = _Store.for_game(GAME)
            for label, counts in reveal_show.items():
                point.record(label, counts)
            n_pairs = _passive_pair_count(sf1, point, omega_bp, fold_idx)
            delta_budget = allocate_simultaneous_delta(
                DELTA,
                public_rows=pilot.num_public_pairs,
                reveal_rows=n_pairs,
            )
            event_entries, event_h, event_meta, n_pairs = _empirical_event_constraints(
                sf1,
                point,
                omega_bp,
                fold_idx,
                delta_budget.reveal_delta,
                METHOD,
            )
        else:
            showdown_reach = agent_showdown_reach(agent_realization, game=GAME)
            n_pairs = _mass_pair_count(sf1, showdown_reach, fold_idx)
            delta_budget = allocate_simultaneous_delta(
                DELTA,
                public_rows=pilot.num_public_pairs,
                reveal_rows=n_pairs,
            )
            event_entries, event_h, event_meta, n_pairs = _empirical_mass_constraints(
                sf1,
                reveal_show,
                showdown_reach,
                fold_idx,
                plan.reveal_budget,
                delta_budget.reveal_delta,
                METHOD,
            )
        assert delta_budget.public_row_delta is not None
        public_intervals = pilot.public_intervals(
            delta_budget.public_row_delta,
            method=METHOD,
            union_bound=False,
        )
        response = _try_solve(
            lambda: robust_safe_response_linear(
                groups,
                public_intervals,
                event_entries,
                event_h,
                v_ref=v_ref,
                eps_safe=RHO,
                game=GAME,
                weights=omega_bp,
                row_meta=event_meta,
            )
        )
        response_source = "empirical_cid"
        if response is None:
            box_infeasible = True
            response = _try_solve(
                lambda: robust_safe_response_public(
                    groups,
                    public_intervals,
                    v_ref=v_ref,
                    eps_safe=RHO,
                    game=GAME,
                    weights=omega_bp,
                )
            )
            response_source = "empirical_cpub"

    if response is None:
        public_infeasible = True
        realization = list(_W["bp_realization"])
        certified = 0.0
        response_source = "blueprint"
    else:
        realization = list(response.realization)
        certified = response.robust_value - v_ref

    y_evaluation = list(Opponent(name=name, behavior=behavior, game=GAME).realization())
    result = {
        **experiment_provenance(),
        "certified": certified,
        "realized": payoff.bilinear(realization, y_evaluation) - v_ref,
        "safety": safety_verifier(realization, game=GAME).value,
        "public_budget": plan.public_budget,
        "reveal_budget": plan.reveal_budget,
        "total_budget": plan.total_budget,
        "public_seed": plan.public_seed,
        "reveal_seed": plan.reveal_seed,
        "protocol": PIPELINE_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "public_pilot_fraction_config": PUBLIC_PILOT_FRACTION,
        "public_fraction": plan.public_fraction,
        "population_inputs_used": mode == "oracle_target",
        "public_set_source": "sampled_blueprint_pilot",
        "route_source": route_source,
        "route_frozen_before_reveal": True,
        "n_route_targets": route_targets,
        "probe_safety": probe_safety,
        "response_source": response_source,
        "n_event_pairs": n_pairs,
        "delta_global": delta_budget.global_delta,
        "delta_public": delta_budget.public_delta,
        "delta_reveal": delta_budget.reveal_delta,
        "public_stat_rows": delta_budget.public_rows,
        "reveal_stat_rows": delta_budget.reveal_rows,
        "public_row_delta": delta_budget.public_row_delta,
        "reveal_row_delta": delta_budget.reveal_row_delta,
        "delta_allocated": delta_budget.allocated_delta,
        "box_infeasible": box_infeasible,
        "pub_infeasible": public_infeasible,
    }
    if mode != "oracle_target":
        assert result["population_inputs_used"] is False
    assert result["public_budget"] + result["reveal_budget"] == total_budget
    return result


_deploy = deploy_cell


def _cell(task: tuple[str, str, int, int]) -> dict[str, Any]:
    """Run one independently reproducible experiment cell."""
    mode, name, episodes, seed = task
    r = _deploy(mode, name, episodes, seed)
    r.update(
        mode=mode,
        opponent=name,
        episodes=episodes,
        seed=seed,
        violation=r["safety"] < _W["floor"] - TOL,
    )
    return r


def _timeout_row(task: tuple[str, str, int, int], reason: str) -> dict[str, Any]:
    """Record a retryable cell that did not reach an independent floor audit."""
    mode, name, episodes, seed = task
    if mode == "oracle":
        public_budget = reveal_budget = 0
        public_seed = reveal_seed = None
    elif mode == "cpub":
        public_budget, reveal_budget = episodes, 0
        public_seed, reveal_seed = 2 * seed, 2 * seed + 1
    else:
        plan = plan_sampling_batches(episodes, seed)
        public_budget, reveal_budget = plan.public_budget, plan.reveal_budget
        public_seed, reveal_seed = plan.public_seed, plan.reveal_seed
    return {
        **experiment_provenance(),
        "mode": mode,
        "opponent": name,
        "episodes": episodes,
        "seed": seed,
        "certified": float("nan"),
        "realized": float("nan"),
        "safety": float("nan"),
        "public_budget": public_budget,
        "reveal_budget": reveal_budget,
        "total_budget": episodes,
        "public_seed": public_seed,
        "reveal_seed": reveal_seed,
        "protocol": PIPELINE_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "public_pilot_fraction_config": PUBLIC_PILOT_FRACTION,
        "public_fraction": (public_budget / episodes if episodes > 0 else 0.0),
        "population_inputs_used": mode in ("oracle", "oracle_target"),
        "box_infeasible": False,
        "cell_failed": True,
        "lp_timeout": not reason.startswith("error:"),
        "execution_error": reason.startswith("error:"),
        "reason": reason,
        "violation": False,
    }


def _load_or_compute_kappa(river_labels: list[str]) -> dict[str, float]:
    """Load or compute kappa for the run safe active de-censoring workflow."""
    path = Path(f"results/kappa_cache_{GAME}_rho{RHO}.json")
    cached: dict[str, float] = {}
    if path.exists():
        cached = {k: float(v) for k, v in json.loads(path.read_text()).items()}
    todo = [lbl for lbl in river_labels if lbl not in cached]
    if not todo:
        return cached
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")

    def _flush() -> None:
        """Persist the accumulated experiment rows to the output file."""
        tmp.write_text(json.dumps(cached))
        tmp.replace(path)

    total = len(river_labels)
    done = len(cached)
    n_timeout = 0
    since_flush = 0
    with ProcessPool(
        max_workers=NPROC,
        initializer=_init,
        context=mp.get_context("spawn"),
    ) as pool:
        futures = {
            pool.schedule(_kappa_task, args=(lbl,), timeout=CELL_TIMEOUT): lbl
            for lbl in todo
        }
        for fut in as_completed(futures):
            lbl = futures[fut]
            try:
                _lbl, k = fut.result()
            except FuturesTimeoutError:
                k, n_timeout = 0.0, n_timeout + 1
            except Exception:
                k = 0.0
            cached[lbl] = k
            done += 1
            since_flush += 1
            if since_flush >= 25:
                _flush()
                since_flush = 0
                print(f"    capacity setup: {done}/{total} LPs", flush=True)
    _flush()
    if n_timeout:
        print(
            f"    capacity setup: {n_timeout} LP(s) hit the per-LP budget "
            f"(recorded kappa=0)",
            flush=True,
        )
    return cached


def main() -> None:
    """Run the command-line entry point."""
    print(
        f"Safe Active De-censoring  game={GAME}  rho={RHO}  N={N_GRID}  "
        f"seeds={len(SEEDS)}  modes={MODES}  ({NPROC} workers)",
        flush=True,
    )

    kappa: dict[str, float] | None = None
    if ABLATION:
        sf1_p = compile_game(GAME, 1)
        river_labels = [i.label for i in sf1_p.info_sets if _is_river(i.label)]
        kappa = _load_or_compute_kappa(river_labels)
        n_pos = sum(1 for v in kappa.values() if v > 1e-9)
        print(
            f"  capacity setup: cached for {len(kappa)} targets ({n_pos} with kappa>0)",
            flush=True,
        )

    print(
        f"  protocol: {PIPELINE_VERSION}; public pilot fraction="
        f"{PUBLIC_PILOT_FRACTION:.2f}",
        flush=True,
    )

    tasks: list[tuple[str, str, int, int]] = []
    for name in LEAKS:
        for mode in MODES:
            if mode == "oracle":
                tasks.append((mode, name, 0, 0))
            else:
                for n in N_GRID:
                    for s in SEEDS:
                        tasks.append((mode, name, n, s))

    out_path = Path(os.environ.get("SAD_OUT", f"results/sad_deploy_e2e_{GAME}.json"))
    ckpt = out_path.with_suffix(".cells.jsonl")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    done_keys: set[tuple[str, str, int, int]] = set()
    ignored_checkpoint_rows = 0
    retry_checkpoint_rows = 0
    if ckpt.exists():
        allow_legacy_config = False
        if os.environ.get("SAD_MIGRATE_LEGACY_CHECKPOINT") == "1" and out_path.exists():
            try:
                legacy_metadata = json.loads(out_path.read_text())
            except (json.JSONDecodeError, OSError):
                legacy_metadata = {}
            allow_legacy_config = legacy_metadata_matches_current(legacy_metadata)
        records = [
            json.loads(line) for line in ckpt.read_text().splitlines() if line.strip()
        ]
        rows, ignored_checkpoint_rows, retry_checkpoint_rows = select_checkpoint_rows(
            records,
            tasks,
            allow_legacy_config=allow_legacy_config,
        )
        ckpt.write_text("".join(json.dumps(row) + "\n" for row in rows))
        done_keys = {
            (r["mode"], r["opponent"], int(r["episodes"]), int(r["seed"])) for r in rows
        }
        print(f"  resuming: {len(done_keys)} cells already checkpointed", flush=True)
        if ignored_checkpoint_rows:
            print(
                f"  ignored {ignored_checkpoint_rows} incompatible checkpoint rows",
                flush=True,
            )
        if retry_checkpoint_rows:
            print(
                f"  retrying {retry_checkpoint_rows} timeout or error rows",
                flush=True,
            )
    pending = [t for t in tasks if t not in done_keys]

    cell_timeout = CELL_TIMEOUT
    total = len(tasks)
    done = len(rows)

    ckpt_fh = open(ckpt, "a")
    with ProcessPool(
        max_workers=NPROC,
        initializer=_init,
        initargs=(None, kappa),
        context=mp.get_context("spawn"),
    ) as pool:
        futures = {
            pool.schedule(_cell, args=(t,), timeout=cell_timeout): t for t in pending
        }
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                r = fut.result()
            except FuturesTimeoutError:
                r = _timeout_row(task, "lp_timeout")
            except Exception as exc:
                r = _timeout_row(task, f"error:{type(exc).__name__}")
            rows.append(r)
            ckpt_fh.write(json.dumps(r) + "\n")
            ckpt_fh.flush()
            done += 1
            if done % 25 == 0:
                print(f"    ... {done}/{total} cells", flush=True)
    ckpt_fh.close()

    cpub = {
        (r["opponent"], int(r["episodes"]), int(r["seed"])): r
        for r in rows
        if r["mode"] == "cpub" and not cell_failed(r)
    }
    oracle = {r["opponent"]: r for r in rows if r["mode"] == "oracle"}

    for row in rows:
        if not cell_failed(row):
            row["capture"] = matched_capture(row, cpub, oracle)

    summary: dict[str, Any] = {}
    print(
        f"\n  {'opponent':<20}{'mode':<14}{'N':>7}{'cert':>8}{'real':>8}{'capture':>8}{'viol':>6}{'infeas':>6}",
        flush=True,
    )
    for name in LEAKS:
        for mode in MODES:
            sub = [r for r in rows if r["opponent"] == name and r["mode"] == mode]
            if not sub:
                continue
            by_n: dict[int, list[dict[str, Any]]] = {}
            for r in sub:
                by_n.setdefault(r["episodes"], []).append(r)
            for n, group in sorted(by_n.items()):
                ok = [r for r in group if not cell_failed(r)]
                n_to = sum(bool(r.get("lp_timeout")) for r in group)
                cert = (
                    statistics.mean(r["certified"] for r in ok) if ok else float("nan")
                )
                real = (
                    statistics.mean(r["realized"] for r in ok) if ok else float("nan")
                )
                viol = sum(bool(r.get("violation")) for r in ok)
                infeas = sum(r.get("box_infeasible", False) for r in ok)
                captures = [r["capture"] for r in ok if not math.isnan(r["capture"])]
                capture = statistics.mean(captures) if captures else float("nan")
                summary[f"{name}|{mode}|{n}"] = {
                    "certified": cert,
                    "realized": real,
                    "capture": capture,
                    "violations": viol,
                    "box_infeasible": infeas,
                    "lp_timeout": n_to,
                    "execution_errors": sum(
                        bool(r.get("execution_error")) for r in group
                    ),
                    "n_cells": len(group),
                }
                inf_str = (
                    f"{infeas:>4}/{n_to:<3}"
                    if mode not in ("cpub", "oracle")
                    else "   .   "
                )
                print(
                    f"  {name:<20}{mode:<14}{n:>7}{cert:>+8.3f}{real:>+8.3f}{capture:>8.2f}{viol:>6} {inf_str}",
                    flush=True,
                )

    completed_rows = [r for r in rows if not cell_failed(r)]
    total_viol = sum(bool(r.get("violation")) for r in completed_rows)
    total_infeas = sum(r.get("box_infeasible", False) for r in rows)
    total_timeout = sum(bool(r.get("lp_timeout")) for r in rows)
    total_errors = sum(bool(r.get("execution_error")) for r in rows)
    print(
        f"\n  TOTAL floor violations across {len(completed_rows)} completed cells = {total_viol}"
        f"  (must be 0 -- structural by Theorem 1)",
        flush=True,
    )
    print(
        f"  TOTAL box infeasibilities (C_pub ∩ C_event empty) = {total_infeas}"
        f"  (deployer falls back to empirical C_pub, then blueprint)",
        flush=True,
    )
    if total_errors:
        print(
            f"  TOTAL execution-error cells = {total_errors}"
            "  (inspect their reasons and rerun before reporting)",
            flush=True,
        )
    print(
        f"  TOTAL lp_timeout cells (degenerate robust LP abandoned) = {total_timeout}"
        f"  (excluded from value means; raise SAD_CELL_TIMEOUT to reduce)",
        flush=True,
    )

    out = out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "game": GAME,
                "rho": RHO,
                "n_grid": N_GRID,
                "seeds": SEEDS,
                "delta": DELTA,
                "protocol": PIPELINE_VERSION,
                "experiment_config": EXPERIMENT_CONFIG,
                "config_fingerprint": CONFIG_FINGERPRINT,
                "public_pilot_fraction": PUBLIC_PILOT_FRACTION,
                "budget_semantics": "episodes = public_budget + reveal_budget",
                "method": METHOD,
                "modes": MODES,
                "leaks": LEAKS,
                "total_violations": total_viol,
                "total_box_infeasibilities": total_infeas,
                "total_lp_timeouts": total_timeout,
                "total_execution_errors": total_errors,
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
