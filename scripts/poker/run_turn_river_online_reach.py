"""Run the turn river online reach experiment. See Experiments and supplementary Certification at the Unbucketed River."""

import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

WAIT_FOR = Path(
    os.environ.get("WAIT_FOR", "results/turn_river_method_table_holdem_tr_b4_25s.json")
)
SEEDS = [
    int(x)
    for x in os.environ.get(
        "OR_REP_SEEDS", ",".join(str(s) for s in range(2026, 2026 + 25))
    ).split(",")
]
WORKERS = int(os.environ.get("WORKERS", "10"))
ROUNDS = int(os.environ.get("OR_ROUNDS", "40"))
EPISODES = int(os.environ.get("OR_EPISODES", "3000"))
SEED_ROUNDS = int(os.environ.get("OR_SEED_ROUNDS", "10"))
GAME = os.environ.get("OR_GAME", "holdem_tr_b4")
OPPONENT = os.environ.get("OR_OPP", "tr_river_overfold_strong")
OUT_DIR = Path(os.environ.get("OR_REP_DIR", "results/turn_river_online_reach_reps"))
LOG_DIR = Path(os.environ.get("OR_REP_LOG_DIR", "logs/turn_river_online_reach_reps"))
OUT = Path(
    os.environ.get(
        "OR_REP_OUT", "results/turn_river_online_reach_replicated_holdem_tr_b4.json"
    )
)


def _wait_for_json(path: Path) -> None:
    """Compute wait for JSON for the run turn river online reach workflow."""
    print(f"waiting for upstream artifact: {path}", flush=True)
    while True:
        if path.exists() and path.stat().st_size > 0:
            try:
                json.load(path.open())
                print(
                    f"upstream ready: {path} ({path.stat().st_size} bytes)", flush=True
                )
                return
            except Exception as exc:
                print(
                    f"  artifact exists but not parseable yet ({exc}); waiting...",
                    flush=True,
                )
        time.sleep(60)


def _run_seed(seed: int) -> dict:
    """Run the seed experiment for the run turn river online reach workflow."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"online_reach_seed_{seed}.json"
    log_path = LOG_DIR / f"online_reach_seed_{seed}.log"
    env = os.environ.copy()
    env.update(
        {
            "OR_SEED": str(seed),
            "OR_OUT": str(out_path),
            "OR_ROUNDS": str(ROUNDS),
            "OR_EPISODES": str(EPISODES),
            "OR_SEED_ROUNDS": str(SEED_ROUNDS),
            "OR_GAME": GAME,
            "OR_OPP": OPPONENT,
        }
    )
    t0 = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(
            [sys.executable, "-u", "-m", "scripts.poker.run_online_reach"],
            cwd=Path.cwd(),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        return {
            "seed": seed,
            "ok": False,
            "returncode": proc.returncode,
            "log": str(log_path),
            "wall_s": time.time() - t0,
        }
    d = json.load(out_path.open())
    summary: dict[str, dict] = {}
    for arm, hist in d["history"].items():
        tail = hist[-5:]
        certs = [h["cert_gain"] for h in tail if h.get("cert_gain") is not None]
        summary[arm] = {
            "last5_realized": sum(h["realized"] for h in tail) / len(tail),
            "last5_cert": (sum(certs) / len(certs)) if certs else None,
            "final_exploit_reach": hist[-1]["exploit_reach"],
            "min_safety": min(h["min_safety"] for h in hist),
            "final_solver": hist[-1].get("solver"),
            "final_box_scale": hist[-1].get("box_scale"),
        }
    return {
        "seed": seed,
        "ok": True,
        "out": str(out_path),
        "log": str(log_path),
        "wall_s": time.time() - t0,
        "oracle_gain": d["oracle_gain"],
        "summary": summary,
    }


def _mean(xs: list[float]) -> float:
    """Compute the arithmetic mean of the supplied observations."""
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    """Run the command-line entry point."""
    _wait_for_json(WAIT_FOR)
    print(
        f"launching online reach reps: {len(SEEDS)} seeds, workers={WORKERS}, "
        f"rounds={ROUNDS}, episodes={EPISODES}, seed_rounds={SEED_ROUNDS}",
        flush=True,
    )
    ctx = mp.get_context("spawn")
    results: list[dict] = []
    with ctx.Pool(processes=min(WORKERS, len(SEEDS))) as pool:
        for r in pool.imap_unordered(_run_seed, SEEDS):
            results.append(r)
            if r.get("ok"):
                obs_seed = r["summary"]["obs_seed"]
                print(
                    f"  seed {r['seed']} ok  obs_seed real={obs_seed['last5_realized']:+.3f} "
                    f"cert={obs_seed['last5_cert']} reach={obs_seed['final_exploit_reach']:.2f} "
                    f"({r['wall_s']:.0f}s)",
                    flush=True,
                )
            else:
                print(
                    f"  seed {r['seed']} FAILED rc={r['returncode']} log={r['log']}",
                    flush=True,
                )

    ok = [r for r in results if r.get("ok")]
    arms = ["core", "obs", "em", "obs_seed"]
    aggregate: dict[str, dict] = {}
    for arm in arms:
        real = [r["summary"][arm]["last5_realized"] for r in ok]
        cert = [
            r["summary"][arm]["last5_cert"]
            for r in ok
            if r["summary"][arm]["last5_cert"] is not None
        ]
        reach = [r["summary"][arm]["final_exploit_reach"] for r in ok]
        mins = [r["summary"][arm]["min_safety"] for r in ok]
        aggregate[arm] = {
            "last5_realized_mean": _mean(real),
            "last5_realized_min": min(real) if real else None,
            "last5_realized_max": max(real) if real else None,
            "last5_cert_mean": _mean(cert) if cert else None,
            "final_exploit_reach_mean": _mean(reach),
            "worst_min_safety": min(mins) if mins else None,
        }

    out = {
        "wait_for": str(WAIT_FOR),
        "game": GAME,
        "opponent": OPPONENT,
        "seeds": SEEDS,
        "rounds": ROUNDS,
        "episodes": EPISODES,
        "seed_rounds": SEED_ROUNDS,
        "workers": WORKERS,
        "results": sorted(results, key=lambda r: r["seed"]),
        "aggregate": aggregate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("\n=== aggregate ===", flush=True)
    for arm in arms:
        a = aggregate[arm]
        print(
            f"{arm:<9} real={a['last5_realized_mean']:+.3f} "
            f"cert={a['last5_cert_mean']} reach={a['final_exploit_reach_mean']:.2f} "
            f"worstMinS={a['worst_min_safety']:+.3f}",
            flush=True,
        )
    print(f"wrote {OUT}", flush=True)
    if len(ok) != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
