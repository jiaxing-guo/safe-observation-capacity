""

import sys

sys.argv = [
    "run_safe_active_decensoring.py",
    "holdem_tr_b2",
    "0.5",
    "10000",
    "1",
    "1",
    "600",
]
from safe_observation.opponents import Opponent
from scripts.poker import run_safe_active_decensoring as D

GAME = "holdem_tr_b2"
DEV = ["river_overfold_w80", "turn_overfold_w70", "revealed_call_strong"]


def _leverage() -> list[float]:
    ""
    bp = D.solve_blueprint(GAME, method="lp")
    atx = D._W["payoff"].matvec_at_x(list(bp.realization))
    return [abs(v) for v in atx]


def _value_anomaly_weights(y_ref: list[float], lev: list[float]) -> dict[str, float]:
    ""
    sf1, omega_bp, y_eq = D._W["sf1"], D._W["omega_bp"], D._W["y_eq"]
    by_hist: dict[str, list] = {}
    for info in sf1.info_sets:
        h = info.label.split("|", 1)[1] if "|" in info.label else ""
        by_hist.setdefault(h, []).append(info)
    val_anom: dict[str, float] = {}
    for h, infos in by_hist.items():
        nA = len(infos[0].children)
        num_ref = [0.0] * nA
        num_eq = [0.0] * nA
        lev_a = [0.0] * nA
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
                lev_a[a] += w * lev[child]
        if den_ref <= 1e-12 or den_eq <= 1e-12:
            continue

        score = sum(
            abs(num_ref[a] / den_ref - num_eq[a] / den_eq) * (lev_a[a] / den_ref)
            for a in range(nA)
        )
        if score > 1e-9:
            val_anom[h] = score
    weights: dict[str, float] = {}
    for info in sf1.info_sets:
        h = info.label.split("|", 1)[1] if "|" in info.label else ""
        if "/" not in h:
            continue
        s = sum(v for hk, v in val_anom.items() if h.startswith(hk))
        if s > 1e-12:
            weights[info.label] = s
    return weights


def _target_leverage(lev: list[float]) -> dict[str, float]:
    ""
    sf1, fold_idx = D._W["sf1"], D._W["fold_idx"]
    out: dict[str, float] = {}
    for info in sf1.info_sets:
        if not D._is_river(info.label):
            continue
        fi = fold_idx.get(info.label)
        s = sum(
            lev[child]
            for a, (_ac, child) in enumerate(info.children)
            if not (fi is not None and a == fi)
        )
        if s > 1e-12:
            out[info.label] = s
    return out


def _reach(weights: dict[str, float]) -> dict[str, float]:
    ""
    if not weights:
        return {}
    beh = D._build_probe_behavior(weights)
    x = D._W["sf0"].realization_from_behavior(beh)
    return D.opponent_reach_weights(x, game=GAME)


def _spearman(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) | set(b))
    if len(keys) < 3:
        return float("nan")

    def ranks(d):
        vals = [d.get(k, 0.0) for k in keys]
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        for rank, i in enumerate(order):
            r[i] = rank
        return r

    ra, rb = ranks(a), ranks(b)
    n = len(keys)
    mra, mrb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - mra) * (rb[i] - mrb) for i in range(n))
    va = sum((ra[i] - mra) ** 2 for i in range(n)) ** 0.5
    vb = sum((rb[i] - mrb) ** 2 for i in range(n)) ** 0.5
    return cov / (va * vb) if va > 0 and vb > 0 else float("nan")


def main() -> None:
    D._init()
    lev = _leverage()
    tlev = _target_leverage(lev)
    print(f"# Value-proxy smoke on {GAME}, rho=0.5\n")
    print(
        f"{'family':<22}{'proxy':<16}{'reached':>9}{'/oracle':>9}"
        f"{'spearman':>10}{'n_tgt':>7}"
    )
    for fam in DEV:
        beh = D._W["suite"][fam]
        y_star = list(Opponent(name=fam, behavior=beh, game=GAME).realization())
        value = D._leak_weights(y_star)
        total_val = sum(value.values())
        dpub = D._public_anomaly_weights(y_star)
        valan = _value_anomaly_weights(y_star, lev)
        dpub_lev = {i: dpub[i] * tlev.get(i, 0.0) for i in dpub if i in tlev}
        cands = {
            "oracle": value,
            "D_pub": dpub,
            "D_pub*lev": dpub_lev,
            "value-anomaly": valan,
        }

        reached: dict[str, float] = {}
        for nm, w in cands.items():
            rch = _reach(w)
            reached[nm] = sum(value.get(i, 0.0) * rch.get(i, 0.0) for i in value)
        ceil = reached["oracle"] or 1.0
        for nm, w in cands.items():
            sp = _spearman(w, value)
            print(
                f"{fam:<22}{nm:<16}{reached[nm]:>9.4f}{reached[nm] / ceil:>9.3f}"
                f"{sp:>10.3f}{len(w):>7d}"
            )
        print(f"{'':<22}{'(total leak value':<16}{total_val:>9.4f})")
        print()


if __name__ == "__main__":
    main()
