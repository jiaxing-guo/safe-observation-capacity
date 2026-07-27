"""Check capacity sensitivity. See supplementary Reproducibility for its role in the release workflow."""

import json
import multiprocessing as mp
from pathlib import Path
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

from concurrent.futures import as_completed

from pebble import ProcessPool

from safe_observation.opponents import Opponent, holdem_equilibrium_opponent
from scripts.poker import run_safe_active_decensoring as D
from scripts.poker.run_turn_river_methods import _perturb, _top_rank

GAME = "holdem_tr_b2"
RHOS = [0.05, 0.1, 0.5]
LP_TIMEOUT = 90.0
CACHE = Path("results/smoke_kbite_cache.jsonl")


CHEAP_LINES = ["cc/7sc", "cc/Ahc"]
EXPENSIVE_LINES = ["cpac/7sc", "cpac/Ahc", "cppac/7sc"]
TURN_DEEP = ["cpp", "cpa"]


def _kappa_at(label: str, rho: float) -> float:
    """Compute kappa at for the check capacity sensitivity workflow."""
    pr = D._try_solve(
        lambda: D.robust_safe_response_probe(
            D._W["triv_iv"],
            D._W["cont_beh"],
            {label: 1.0},
            v_ref=D._W["v_ref"],
            eps_safe=rho,
            beta=D.BETA,
            rho=0.0,
            game=GAME,
        )
    )
    if pr is None:
        return 0.0
    x = list(pr.realization)
    return float(D.opponent_reach_weights(x, game=GAME).get(label, 0.0))


def _ktask(args):
    """Compute ktask for the check capacity sensitivity workflow."""
    label, rho = args
    return label, rho, _kappa_at(label, rho)


def _probe_reach(weights: dict[str, float], rho: float) -> dict[str, float]:
    """Compute probe reach for the check capacity sensitivity workflow."""
    if not weights:
        return {}
    pr = D._try_solve(
        lambda: D.robust_safe_response_probe(
            D._W["triv_iv"],
            D._W["cont_beh"],
            weights,
            v_ref=D._W["v_ref"],
            eps_safe=rho,
            beta=D.BETA,
            rho=0.0,
            game=GAME,
        )
    )
    if pr is None:
        return {}
    return D.opponent_reach_weights(list(pr.realization), game=GAME)


def _build_leak(actions):
    """Build leak for the check capacity sensitivity workflow."""
    eq = holdem_equilibrium_opponent(GAME).behavior
    cheap = set(CHEAP_LINES)
    deep = set(EXPENSIVE_LINES) | set(TURN_DEEP)

    def band(hole):
        """Compute or draw a mean-and-uncertainty band."""
        return _top_rank(hole) >= 11

    def cheap_call(hole, hist, acts):
        """Compute cheap call for the check capacity sensitivity workflow."""
        return hist in cheap and band(hole) and "c" in acts

    def deep_fold(hole, hist, acts):
        """Compute deep fold for the check capacity sensitivity workflow."""
        return hist in deep and band(hole) and "f" in acts

    y = _perturb(eq, actions, cheap_call, "c", 0.6)
    y = _perturb(y, actions, deep_fold, "f", 0.7)
    return y


def _load_cache() -> dict:
    """Load cache for the check capacity sensitivity workflow."""
    out = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["key"]] = r["val"]
    return out


def _append_cache(key: str, val) -> None:
    """Compute append cache for the check capacity sensitivity workflow."""
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a") as fh:
        fh.write(json.dumps({"key": key, "val": val}) + "\n")


def main() -> None:
    """Run the command-line entry point."""
    D._init()
    actions = {i.label: [a for a, _ in i.children] for i in D._W["sf1"].info_sets}
    beh = _build_leak(actions)
    y_star = list(Opponent(name="conc_line", behavior=beh, game=GAME).realization())
    value = D._leak_weights(y_star)
    dpub = D._public_anomaly_weights(y_star)

    tot = sum(value.values()) or 1.0
    top5 = sum(sorted(value.values(), reverse=True)[:5]) / tot
    states = {lbl.split("|", 1)[1] for lbl in value}
    print(f"# line-concentrated capacity sensitivity on {GAME}")
    print(
        f"leak: {len(value)} value targets, top5 share {top5:.3f}, "
        f"{len(states)} public states\n"
    )

    cache = _load_cache()
    targets = list(value)
    jobs = [(lbl, r) for lbl in targets for r in RHOS if f"k|{lbl}|{r}" not in cache]
    cached_n = len(targets) * len(RHOS) - len(jobs)
    print(f"kappa LPs: {len(jobs)} to compute, {cached_n} cached")
    if jobs:
        ctx = mp.get_context("spawn")
        with ProcessPool(max_workers=12, context=ctx, initializer=D._init) as pool:
            futs = {
                pool.schedule(_ktask, args=(j,), timeout=LP_TIMEOUT): j for j in jobs
            }
            done = 0
            for fut in as_completed(futs):
                lbl, rho_j = futs[fut][0], futs[fut][1]
                try:
                    _, _, k = fut.result()
                except Exception:
                    k = 0.0
                _append_cache(f"k|{lbl}|{rho_j}", k)
                cache[f"k|{lbl}|{rho_j}"] = k
                done += 1
                if done % 20 == 0:
                    print(f"  kappa {done}/{len(jobs)}")

    print(
        f"\n{'rho':>6}{'kappaCV':>9}{'oracle':>9}{'D_pub':>9}{'Dpub*kap':>10}{'lift%':>8}"
    )
    for r in RHOS:
        kappa = {t: cache.get(f"k|{t}|{r}", 0.0) for t in targets}
        nz = [k for k in kappa.values() if k > 1e-12]
        cv = (st.pstdev(nz) / st.mean(nz)) if len(nz) > 1 and st.mean(nz) > 0 else 0.0
        dpub_kap = {
            i: dpub[i] * kappa.get(i, 0.0) for i in dpub if kappa.get(i, 0.0) > 1e-12
        }
        reached = {}
        for wnm, w in (("oracle", value), ("dpub", dpub), ("dpub_kap", dpub_kap)):
            ckey = f"reach|{wnm}|{r}"
            if ckey in cache:
                rch = cache[ckey]
            else:
                rch = _probe_reach(w, r)
                _append_cache(ckey, rch)
                cache[ckey] = rch
            reached[wnm] = sum(value.get(i, 0.0) * rch.get(i, 0.0) for i in value)
        base = reached["dpub"] or 1e-12
        lift = 100.0 * (reached["dpub_kap"] - reached["dpub"]) / base
        oc = reached["oracle"] or 1.0
        print(
            f"{r:>6.2f}{cv:>9.2f}{reached['oracle'] / oc:>9.3f}"
            f"{reached['dpub'] / oc:>9.3f}{reached['dpub_kap'] / oc:>10.3f}{lift:>8.1f}"
        )
    print("\nBITE if Dpub*kap > D_pub (positive lift%) at small rho.")


if __name__ == "__main__":
    main()
