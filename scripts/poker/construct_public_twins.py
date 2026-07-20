""

from __future__ import annotations

from collections import defaultdict
import sys

from safe_observation.opponents import Opponent, holdem_equilibrium_opponent
from safe_observation.payoff import build as build_payoff
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import (
    safety_constrained_best_response,
    solve_blueprint,
)

GAME = sys.argv[1] if len(sys.argv) > 1 else "holdem_tr_b2"
DELTAS = [0.05, 0.10]
RHOS = [0.1, 0.3, 0.5]
MIN_MASS = 0.20
MAX_RATIO = 4.0
MIN_GAP = 0.15


def _infos_by_history(sf):
    by = defaultdict(list)
    for info in sf.info_sets:
        _hand, hist = info.label.split("|", 1)
        acts = [a for a, _ in info.children]
        if "f" in acts and "c" in acts:
            by[hist].append(info)
    return by


def _pick_targets(sf, eq, yr, max_targets=4):
    ""
    by = _infos_by_history(sf)
    cands = []
    for hist, infos in by.items():
        rich = []
        for info in infos:
            m = yr[info.parent_seq]
            acts = [a for a, _ in info.children]
            fi = acts.index("f")
            fr = eq[info.label][fi]
            if m > MIN_MASS and 0.10 < fr < 0.90:
                rich.append((info, m, fr))
        if len(rich) < 2:
            continue

        rich.sort(key=lambda z: z[2])
        lo, hi = rich[0], rich[-1]
        if hi[2] - lo[2] < MIN_GAP:
            continue
        m1, m2 = hi[1], lo[1]
        if max(m1, m2) / min(m1, m2) > MAX_RATIO:
            continue
        depth = hist.count("/")
        cands.append((hi[2] - lo[2], depth, hist, hi[0], hi[1], lo[0], lo[1]))

    cands.sort(key=lambda z: (z[1], -z[0]))
    return cands


def _perturbed(eq, info1, m1, info2, m2, eps, sign):
    ""
    beh = {k: list(v) for k, v in eq.items()}
    for info, m, s in ((info1, m1, +1.0), (info2, m2, -1.0)):
        acts = [a for a, _ in info.children]
        fi, ci = acts.index("f"), acts.index("c")
        d = sign * s * eps / m
        row = beh[info.label]
        if not (0.0 <= row[fi] + d <= 1.0 and 0.0 <= row[ci] - d <= 1.0):
            return None
        row[fi] += d
        row[ci] -= d
    return beh


def _tv_public(sf, yr_a, yr_b):
    ""

    def summed(yr):
        out = defaultdict(float)
        for info in sf.info_sets:
            _hand, hist = info.label.split("|", 1)
            for a, (_ac, child) in enumerate(info.children):
                out[(hist, a)] += yr[child]
        return out

    sa, sb = summed(yr_a), summed(yr_b)
    return 0.5 * sum(abs(sa[k] - sb[k]) for k in set(sa) | set(sb))


def main() -> None:
    sf = compile_game(GAME, 1)
    payoff = build_payoff(GAME)
    v_ref = solve_blueprint(GAME, method="lp").value
    eq = holdem_equilibrium_opponent(GAME).behavior
    yr_base = list(Opponent(name="eq", behavior=eq, game=GAME).realization())

    candidates = _pick_targets(sf, eq, yr_base)

    targets, seen_depth = [], set()
    for cand in candidates:
        _gap_fr, depth, _hist, i1, m1, i2, m2 = cand
        eps_probe = 0.05 * min(m1, m2)
        bp = _perturbed(eq, i1, m1, i2, m2, eps_probe, +1.0)
        bm = _perturbed(eq, i1, m1, i2, m2, eps_probe, -1.0)
        if bp is None or bm is None:
            continue
        yrp = list(Opponent(name="yp", behavior=bp, game=GAME).realization())
        yrm = list(Opponent(name="ym", behavior=bm, game=GAME).realization())
        if _tv_public(sf, yrp, yrm) > 1e-9:
            continue
        targets.append(((depth, depth in seen_depth), cand))
        seen_depth.add(depth)
    targets.sort(key=lambda z: z[0])
    targets = [c for _k, c in targets][:4]

    print(f"# Hold'em hard-direction kink on {GAME}  (v_ref={v_ref:.4f})")
    print(
        f"# {len(targets)} public-twin targets (tv_pub<1e-9); "
        "eps in mass units; c_eps>0 => non-vacuous bound\n"
    )

    gbar_by_rho = {
        rho: safety_constrained_best_response(
            eq, v_ref=v_ref, eps_safe=rho, game=GAME
        ).value
        - v_ref
        for rho in RHOS
    }

    for _gap_fr, depth, hist, i1, m1, i2, m2 in targets:
        h1 = i1.label.split("|", 1)[0]
        h2 = i2.label.split("|", 1)[0]
        line = "turn" if depth == 0 else "river"
        print(
            f"== target: hist={hist!r} ({line})  hands {h1} (fold {eq[i1.label][[a for a, _ in i1.children].index('f')]:.2f}) "
            f"vs {h2} (fold {eq[i2.label][[a for a, _ in i2.children].index('f')]:.2f})  m=({m1:.3f},{m2:.3f})",
            flush=True,
        )
        print(
            f"{'rho':>5}{'delta':>7}{'eps':>8}{'tv_pub':>9}{'g(ybar)':>9}{'c_eps':>9}{'gap':>8}  regime"
        )
        for rho in RHOS:
            gbar = gbar_by_rho[rho]
            for delta in DELTAS:
                eps = delta * min(m1, m2)
                bp = _perturbed(eq, i1, m1, i2, m2, eps, +1.0)
                bm = _perturbed(eq, i1, m1, i2, m2, eps, -1.0)
                if bp is None or bm is None:
                    print(
                        f"{rho:>5.2f}{delta:>7.2f}    (perturbation leaves [0,1]; skip)"
                    )
                    continue
                yrp = list(Opponent(name="yp", behavior=bp, game=GAME).realization())
                yrm = list(Opponent(name="ym", behavior=bm, game=GAME).realization())
                tv = _tv_public(sf, yrp, yrm)
                rp = safety_constrained_best_response(
                    bp, v_ref=v_ref, eps_safe=rho, game=GAME
                )
                rm = safety_constrained_best_response(
                    bm, v_ref=v_ref, eps_safe=rho, game=GAME
                )
                gp, gm = rp.value - v_ref, rm.value - v_ref
                c_eps = (gp + gm - 2.0 * gbar) / (2.0 * eps)

                v_cross = payoff.bilinear(list(rp.realization), yrm) - v_ref
                gap = gm - v_cross
                regime = "switch (c>0)" if c_eps > 1e-4 else "floor-pinned"
                print(
                    f"{rho:>5.2f}{delta:>7.2f}{eps:>8.4f}{tv:>9.1e}{gbar:>9.4f}{c_eps:>9.4f}"
                    f"{gap:>8.4f}  {regime}",
                    flush=True,
                )
        print(flush=True)


if __name__ == "__main__":
    main()
