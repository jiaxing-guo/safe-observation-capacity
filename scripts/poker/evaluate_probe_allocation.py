""

from __future__ import annotations

from collections import defaultdict
import json
import math
import statistics as st
import sys

from safe_observation.opponents import Opponent, holdem_equilibrium_opponent
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import solve_blueprint
from scripts.poker.run_turn_river_methods import _perturb, _top_rank

GAME = "holdem_tr_b2"
RHO = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
EPS = 0.05
C_CERT = 0.25


def _is_river(label: str) -> bool:
    return "/" in label.split("|", 1)[1]


def _depth_spread_leak(eq, actions):
    ""

    def turn_line(hole, hist, acts):
        return "/" not in hist and "f" in acts and _top_rank(hole) >= 9

    def river_line(hole, hist, acts):
        return "/" in hist and "f" in acts and _top_rank(hole) >= 9

    leak = _perturb(eq, actions, turn_line, "f", 0.6)
    leak = _perturb(leak, actions, river_line, "f", 0.5)
    return leak


def _public_anomaly(sf1, omega_bp, y_ref, y_eq):
    ""
    by_hist = defaultdict(list)
    for info in sf1.info_sets:
        by_hist[info.label.split("|", 1)[1]].append(info)
    anomaly = {}
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
    weights = {}
    for info in sf1.info_sets:
        h = info.label.split("|", 1)[1]
        if "/" not in h:
            continue
        score = sum(tv for hk, tv in anomaly.items() if h.startswith(hk))
        if score > 1e-6:
            weights[info.label] = score
    return weights


def _spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = st.mean(rx), st.mean(ry)
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    sx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    sy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    return cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def _allocate(targets, score_key, budget):
    ""
    ranked = sorted(targets, key=lambda t: t[score_key], reverse=True)
    spent = 0.0
    captured = 0.0
    for t in ranked:
        if t["lambda"] <= 1e-9:
            continue
        cost = C_CERT / (t["lambda"] * EPS * EPS)
        if spent + cost > budget:
            continue
        spent += cost
        captured += t["value"]
    return captured, spent


def main() -> None:
    sf1 = compile_game(GAME, 1)
    bp = solve_blueprint(GAME, method="lp")
    eq = holdem_equilibrium_opponent(GAME).behavior
    actions = {i.label: [a for a, _ in i.children] for i in sf1.info_sets}
    y_eq = list(Opponent(name="eq", behavior=eq, game=GAME).realization())

    from safe_observation.solvers import opponent_reach_weights

    omega_bp = opponent_reach_weights(bp.realization, game=GAME)

    leak = _depth_spread_leak(eq, actions)
    y_leak = list(Opponent(name="leak", behavior=leak, game=GAME).realization())

    kappa = json.load(open(f"results/kappa_cache_{GAME}_rho{RHO}.json"))
    dpub = _public_anomaly(sf1, omega_bp, y_leak, y_eq)

    info_by = {i.label: i for i in sf1.info_sets}
    targets = []
    for lbl, dv in dpub.items():
        k = kappa.get(lbl, 0.0)
        info = info_by[lbl]
        pi = y_eq[info.parent_seq]
        lam = k * pi
        depth = lbl.split("|", 1)[1].count("/")
        targets.append(
            {
                "label": lbl,
                "value": dv,
                "kappa": k,
                "pi": pi,
                "lambda": lam,
                "depth": depth,
                "dv_lambda": dv * lam,
            }
        )
    targets = [t for t in targets if t["lambda"] > 1e-9]
    print(f"# Probe A: depth-spread leak, rho={RHO}, {len(targets)} reveal targets\n")

    lams = [t["lambda"] for t in targets]
    vals = [t["value"] for t in targets]
    print(
        f"lambda: min {min(lams):.5f} median {st.median(lams):.5f} max {max(lams):.5f} "
        f"CV {st.pstdev(lams) / st.mean(lams):.3f}"
    )
    print(
        f"value (D_pub): min {min(vals):.4f} median {st.median(vals):.4f} "
        f"max {max(vals):.4f}"
    )
    print(
        f"spearman(value, lambda) = {_spearman(vals, lams):+.3f}  "
        "(negative => anti-correlated => allocation matters)"
    )

    by_d = defaultdict(list)
    for t in targets:
        by_d[t["depth"]].append(t)
    print("\nby depth: n, mean lambda, mean value")
    for d in sorted(by_d):
        ts = by_d[d]
        print(
            f"  depth {d}: n={len(ts):3d}  mean_lambda={st.mean([x['lambda'] for x in ts]):.5f}  "
            f"mean_value={st.mean([x['value'] for x in ts]):.4f}"
        )

    print("\n=== certified value captured: DV-only vs DV*lambda (Eq 2) ===")
    print(f"{'budget':>10}{'DV-only':>10}{'DV*lam':>10}{'lift%':>8}")
    total_cost = sum(C_CERT / (t["lambda"] * EPS * EPS) for t in targets)
    for frac in (0.1, 0.25, 0.5, 1.0):
        B = frac * total_cost
        v_dv, _ = _allocate(targets, "value", B)
        v_dvl, _ = _allocate(targets, "dv_lambda", B)
        lift = 100 * (v_dvl - v_dv) / v_dv if v_dv > 1e-9 else float("nan")
        print(f"{B:>10.0f}{v_dv:>10.3f}{v_dvl:>10.3f}{lift:>8.1f}")

    out = {
        "game": GAME,
        "rho": RHO,
        "eps": EPS,
        "n_targets": len(targets),
        "spearman_value_lambda": _spearman(vals, lams),
        "lambda_cv": st.pstdev(lams) / st.mean(lams),
        "targets": targets,
    }
    with open(f"results/probe_allocation_rho{RHO}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n# wrote results/probe_allocation_rho{RHO}.json")


if __name__ == "__main__":
    main()
