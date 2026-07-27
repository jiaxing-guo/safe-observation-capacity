"""Check the signals and reach rates used by the acquisition router."""

import multiprocessing as mp
import statistics as st
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

DEV = ["river_overfold_w80", "turn_overfold_w70", "revealed_call_strong"]
RHOS = [0.05, 0.1, 0.5]
TOPK = 15


def _kappa_at(label: str, rho: float) -> float:
    """Compute kappa at for the check router mechanism workflow."""
    pr = D._try_solve(
        lambda: D.robust_safe_response_probe(
            D._W["triv_iv"],
            D._W["cont_beh"],
            {label: 1.0},
            v_ref=D._W["v_ref"],
            eps_safe=rho,
            beta=D.BETA,
            rho=0.0,
            game="holdem_tr_b2",
        )
    )
    if pr is None:
        return 0.0
    x = list(pr.realization)
    return float(D.opponent_reach_weights(x, game="holdem_tr_b2").get(label, 0.0))


def _task(args):
    """Execute one independently reproducible parallel task."""
    fam, label = args
    return fam, label, {r: _kappa_at(label, r) for r in RHOS}


def main() -> None:
    """Run the command-line entry point."""
    D._init()
    suite = D._W["suite"]

    fam_dpub, fam_leak, candidates = {}, {}, []
    print("# Router mechanism smokes (holdem_tr_b2)\n")
    print("=== SMOKE 1: does D_pub point where the value is? (no LP) ===")
    for fam in DEV:
        beh = suite[fam]
        y_star = list(
            Opponent(name=fam, behavior=beh, game="holdem_tr_b2").realization()
        )
        dpub = D._public_anomaly_weights(y_star)
        leak = D._leak_weights(y_star)
        fam_dpub[fam], fam_leak[fam] = dpub, leak
        dtop = [lab for lab, _ in sorted(dpub.items(), key=lambda kv: -kv[1])[:TOPK]]
        ltop = [lab for lab, _ in sorted(leak.items(), key=lambda kv: -kv[1])[:TOPK]]
        overlap = len(set(dtop) & set(ltop))

        leak_tot = sum(leak.values()) or 1.0
        dpub_cov = sum(leak.get(lab, 0.0) for lab in dtop) / leak_tot
        leak_cov = sum(leak.get(lab, 0.0) for lab in ltop) / leak_tot
        print(
            f"  {fam:20s} |Dpub|={len(dpub):4d} |leak|={len(leak):4d}  "
            f"top{TOPK} overlap={overlap:2d}/{TOPK}  "
            f"leak-value covered by Dpub-top={dpub_cov:.2f} (leak-top={leak_cov:.2f})"
        )
        for lbl in set(dtop) | set(ltop):
            candidates.append((fam, lbl))

    print(
        f"\n=== SMOKE 2: capacity spread among value targets ({len(candidates)} targets) ==="
    )
    print(
        "    (if kappa varies among value targets at small rho, DV*kappa can reorder)"
    )
    results: dict[str, dict] = {}
    with mp.Pool(12, initializer=D._init) as pool:
        for fam, label, kap in pool.imap_unordered(_task, candidates, chunksize=2):
            results.setdefault(fam, {})[label] = kap

    for fam in DEV:
        leak = fam_leak[fam]

        val = {lab: k for lab, k in results[fam].items() if leak.get(lab, 0.0) > 1e-9}
        print(f"\n  {fam}")
        for r in RHOS:
            ks = [k[r] for k in val.values()]
            nz = [k for k in ks if k > 1e-9]
            if not ks:
                continue
            cv = (
                (st.pstdev(nz) / st.mean(nz))
                if len(nz) > 1 and st.mean(nz) > 0
                else 0.0
            )
            print(
                f"    rho={r:<4}  kappa over {len(ks):2d} value targets: "
                f"reachable={len(nz):2d}/{len(ks):2d}  "
                f"min={min(ks):.4f} med={st.median(ks):.4f} max={max(ks):.4f}  CV={cv:.2f}"
            )

        topval = sorted(val.items(), key=lambda kv: -leak[kv[0]])[:6]
        k01 = [k[0.1] for _, k in topval]
        if k01:
            spread = (max(k01) - min(k01)) / (max(k01) + 1e-12)
            print(
                f"    -> top-6 value targets kappa@0.1: {[round(x, 4) for x in k01]}  "
                f"spread={spread:.2f} (0=uniform/inert, 1=wide/reorders)"
            )


if __name__ == "__main__":
    main()
