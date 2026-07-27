"""Evaluate the probe allocation bound experiment. See Experiments and supplementary Certification at the Unbucketed River."""

from __future__ import annotations

from collections import defaultdict
import json
import statistics as st
import sys

from safe_observation.opponents import Opponent
from safe_observation.payoff import build as build_payoff
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import (
    agent_showdown_reach,
    robust_safe_response_probe,
    safety_constrained_best_response,
    solve_blueprint,
)

GAME = sys.argv[1] if len(sys.argv) > 1 else "leduc"
EPS = [0.2, 0.1, 0.05]
RHO = 0.3


def _facing_bet_targets(sf1):
    """Compute facing bet targets for the evaluate probe allocation bound workflow."""
    fold_chars = {"p", "f"}
    call_chars = {"b", "c"}
    out = {}
    for info in sf1.info_sets:
        acts = [a for a, _ in info.children]
        fi = next((i for i, a in enumerate(acts) if a in fold_chars), None)
        ci = next((i for i, a in enumerate(acts) if a in call_chars), None)
        if fi is not None and ci is not None:
            out[info.label] = (fi, ci)
    return out


def _real(beh):
    """Compute real for the evaluate probe allocation bound workflow."""
    return list(Opponent(name="x", behavior=beh, game=GAME).realization())


def _base_behavior(sf1):
    """Compute base behavior for the evaluate probe allocation bound workflow."""
    return {i.label: [1.0 / len(i.children)] * len(i.children) for i in sf1.info_sets}


def _public_key(label: str) -> str:
    """Compute public key for the evaluate probe allocation bound workflow."""
    if ":" in label and "|" not in label:
        return label.split(":", 1)[1]
    if "|" in label:
        return label.split("|", 1)[1]
    return label


def _reveal_lambda(sf1, label, fi, ci, x_real):
    """Compute reveal lambda for the evaluate probe allocation bound workflow."""
    sd = agent_showdown_reach(x_real, game=GAME)
    row = sd.get(label)
    if row is None or ci >= len(row):
        return 0.0
    w_sd, _committing = row[ci]
    return w_sd


def main() -> None:
    """Run the command-line entry point."""
    sf1 = compile_game(GAME, 1)
    payoff = build_payoff(GAME)
    bp = solve_blueprint(GAME, method="lp")
    v_ref = bp.value
    cand = _facing_bet_targets(sf1)
    base = _base_behavior(sf1)

    cont_beh = {}
    for info in sf1.info_sets:
        lbl = info.label
        if lbl in cand:
            fi, ci = cand[lbl]
            row = [0.0] * len(info.children)
            row[ci] = 1.0
            cont_beh[lbl] = row
        else:
            cont_beh[lbl] = [1.0 / len(info.children)] * len(info.children)
    triv_iv = {i.label: [(0.0, 1.0)] * len(i.children) for i in sf1.info_sets}
    probe = robust_safe_response_probe(
        triv_iv,
        cont_beh,
        {lbl: 1.0 for lbl in cand},
        v_ref=v_ref,
        eps_safe=RHO,
        beta=1e6,
        rho=0.0,
        game=GAME,
    )
    x_probe = list(probe.realization)

    targets = []
    for lbl, (fi, ci) in cand.items():
        lam = _reveal_lambda(sf1, lbl, fi, ci, x_probe)
        if lam > 1e-6:
            targets.append(
                {
                    "label": lbl,
                    "fi": fi,
                    "ci": ci,
                    "lambda": lam,
                    "pub": _public_key(lbl),
                }
            )
    targets.sort(key=lambda t: -t["lambda"])
    print(f"# Probe B: allocation lower bound on {GAME} (v_ref={v_ref:.4f}, rho={RHO})")
    print(f"# {len(targets)} facing-bet reveal targets with lambda>0\n")

    pubs = defaultdict(list)
    for t in targets:
        pubs[t["pub"]].append(t["label"])
    print(
        f"distinct public reveal states: {len(pubs)} over {len(targets)} targets "
        f"(disjoint reveal events => additive cost)\n"
    )

    lams = [t["lambda"] for t in targets]
    print(
        f"lambda: min {min(lams):.5f} median {st.median(lams):.5f} max {max(lams):.5f} "
        f"CV {st.pstdev(lams) / st.mean(lams):.3f}\n"
    )

    print("=== (R) per-target rate check  N*lam*eps^2 ~ const ===")
    sample = targets[:: max(1, len(targets) // 4)][:4]
    for t in sample:
        lbl, fi, ci = t["label"], t["fi"], t["ci"]
        info = next(i for i in sf1.info_sets if i.label == lbl)
        child_call = info.children[ci][1]
        row_consts = []
        for eps in EPS:
            yb = {k: list(v) for k, v in base.items()}
            r = yb[lbl]
            if not (0.0 <= r[fi] + eps <= 1.0 and 0.0 <= r[ci] - eps <= 1.0):
                continue
            r[fi] += eps
            r[ci] -= eps
            yr = _real(yb)
            yr0 = _real(base)
            d_act = abs(
                _reveal_lambda(sf1, lbl, fi, ci, x_probe)
                * (yr[child_call] - yr0[child_call])
            )
            d_act = max(d_act, 1e-12)
            ncert = t["lambda"] / d_act**2
            row_consts.append(ncert * t["lambda"] * eps**2)
        if row_consts:
            print(
                f"  {lbl:<22} lam={t['lambda']:.5f}  N*lam*e2 over eps: "
                + ", ".join(f"{c:.3f}" for c in row_consts)
            )

    print("\n=== (A) allocation cost to certify all M targets (c=1/4, eps=0.05) ===")
    eps = 0.05
    inv = [1.0 / t["lambda"] for t in targets]
    opt = 0.25 / eps**2 * sum(inv)
    M = len(targets)
    uniform = 0.25 / eps**2 * M * max(inv)
    print(
        f"  M={M}  optimal(Sum 1/lam)={opt:.0f}  uniform(M*max 1/lam)={uniform:.0f}  "
        f"ratio={uniform / opt:.1f}x"
    )

    try:
        kap = json.load(open("results/kappa_cache_holdem_tr_b2_rho0.1.json"))
        hv = sorted(v for v in kap.values() if v > 1e-6)
        invh = [1.0 / v for v in hv]
        Mh = len(hv)
        ratioh = (Mh * max(invh)) / sum(invh)

        print("\n=== hold'em real-capacity allocation (rho=0.1, 2147 targets) ===")
        print(
            f"  full set M={Mh}: uniform/optimal = {ratioh:.1f}x  "
            f"(capacity CV {st.pstdev(hv) / st.mean(hv):.2f})"
        )
        for k in (10, 50, 200):
            sub = sorted(invh)[:k]
            print(f"  cheapest {k}: uniform/optimal = {(k * max(sub)) / sum(sub):.1f}x")
    except FileNotFoundError:
        pass

    out = {
        "game": GAME,
        "rho": RHO,
        "n_targets": len(targets),
        "n_public_states": len(pubs),
        "lambda_cv": st.pstdev(lams) / st.mean(lams),
        "alloc_ratio_uniform_over_opt": uniform / opt,
        "targets": targets,
    }
    with open(f"results/probe_allocation_lb_{GAME}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n# wrote results/probe_allocation_lb_{GAME}.json")
    _ = (payoff, safety_constrained_best_response)


if __name__ == "__main__":
    main()
