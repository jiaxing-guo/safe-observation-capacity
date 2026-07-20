""

import json
from pathlib import Path
import statistics
import sys

GAME = sys.argv[1] if len(sys.argv) > 1 else "holdem_tr_b2"
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
N_GRID = [1000, 10000, 100000, 1000000]
SEEDS = [2026 + i for i in range(N_SEEDS)]
DEV = ["river_overfold_w80", "turn_overfold_w70", "revealed_call_strong"]


sys.argv = ["run_safe_active_decensoring.py", GAME, "0.5", "1000", "1", "1", "600"]
from safe_observation.opponents import Opponent
from scripts.poker import run_safe_active_decensoring as D


def _probe_for(name: str):
    ""
    behavior = D._W["suite"][name]
    y_star = list(Opponent(name=name, behavior=behavior, game=GAME).realization())
    weights = D._public_anomaly_weights(y_star)
    if not weights:
        return D._W["bp_behavior"], y_star
    return D._build_probe_behavior(weights), y_star


def _committing_pins(agent_behavior):
    ""
    sf1, fold_idx = D._W["sf1"], D._W["fold_idx"]
    agent_real = D._W["sf0"].realization_from_behavior(agent_behavior)
    sd_reach = D.agent_showdown_reach(agent_real, game=GAME)
    pins = []
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
                pins.append((info.label, a, child, w_sd))
    return pins


def _passive_pins(y_star):
    ""
    sf1, fold_idx = D._W["sf1"], D._W["fold_idx"]
    bp_real = D._W["sf0"].realization_from_behavior(D._W["bp_behavior"])
    omega = D.opponent_reach_weights(bp_real, game=GAME)
    pins = []
    for info in sf1.info_sets:
        if omega.get(info.label, 0.0) <= 0.0:
            continue
        fi = fold_idx.get(info.label)
        for a, (_act, child) in enumerate(info.children):
            if fi is not None and a == fi:
                continue
            pins.append((info.label, a, child, info.parent_seq))
    return pins


def main() -> None:
    D._init()
    out: dict[str, dict] = {}
    print(f"# Q5 C_id coverage/width  game={GAME}  seeds={N_SEEDS}", flush=True)
    print(
        f"  {'opponent':<22}{'N':>9}{'cov_act':>10}{'cov_pass':>10}"
        f"{'meanwidth':>11}{'pins':>7}",
        flush=True,
    )
    for name in DEV:
        agent_behavior, y_star = _probe_for(name)
        pins = _committing_pins(agent_behavior)
        n_pairs = max(1, len(pins))
        per = D.DELTA / n_pairs
        behavior = D._W["suite"][name]

        ppins = _passive_pins(y_star)
        pper = D.DELTA / max(1, len(ppins))
        out[name] = {"n_pairs": len(pins), "n_passive_pairs": len(ppins), "by_N": {}}
        for N in N_GRID:
            cov_s, wid_s, exc_s, pcov_s = [], [], [], []
            for seed in SEEDS:
                _pay, show, _fold = D.native.simulate_showdown(
                    GAME, agent_behavior, behavior, N, seed
                )
                covered = 0
                widths = []
                worst_excess = 0.0
                for label, a, child, w_sd in pins:
                    cnt = show.get(label, [0] * (a + 1))
                    mhat = (cnt[a] if a < len(cnt) else 0) / N
                    lo, hi = D.empirical_bernstein_interval(mhat, N, per)
                    truth = w_sd * y_star[child]
                    if lo - 1e-12 <= truth <= hi + 1e-12:
                        covered += 1
                    else:
                        worst_excess = max(
                            worst_excess, truth - hi if truth > hi else lo - truth
                        )
                    widths.append(hi - lo)
                cov_s.append(covered / len(pins) if pins else 1.0)
                wid_s.append(statistics.mean(widths) if widths else 0.0)
                exc_s.append(worst_excess)

                _p2, pshow, _f2 = D.native.simulate_showdown(
                    GAME, D._W["bp_behavior"], behavior, N, seed
                )
                pstore = D._Store.for_game(GAME)
                for lab, c in pshow.items():
                    pstore.record(lab, c)
                pcov = 0
                ptot = 0
                for label, a, child, pseq in ppins:
                    if pstore.visits(label) <= 0:
                        continue
                    box = pstore.interval(label, pper, D.METHOD)
                    if a >= len(box):
                        continue
                    lo, hi = box[a]
                    parent = y_star[pseq]
                    v = y_star[child]
                    ptot += 1
                    if lo * parent - 1e-9 <= v <= hi * parent + 1e-9:
                        pcov += 1
                pcov_s.append(pcov / ptot if ptot else 1.0)

            def _ms(xs):
                m = statistics.mean(xs)
                se = statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0
                return m, se

            cm, cse = _ms(cov_s)
            wm, wse = _ms(wid_s)
            em, _ = _ms(exc_s)
            pcm, pcse = _ms(pcov_s)
            out[name]["by_N"][N] = {
                "coverage": cm,
                "coverage_se": cse,
                "passive_coverage": pcm,
                "passive_coverage_se": pcse,
                "mean_width": wm,
                "mean_width_se": wse,
                "worst_excess": em,
                "n_seeds": len(SEEDS),
            }
            print(
                f"  {name:<22}{N:>9}{cm:>10.3f}{pcm:>10.3f}{wm:>11.3e}{len(pins):>7}",
                flush=True,
            )

    path = Path(f"results/cid_coverage_{GAME}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {path}", flush=True)


if __name__ == "__main__":
    main()
