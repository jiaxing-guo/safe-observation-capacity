""

import json
from pathlib import Path

from safe_observation.experiments.online import run_showdown_comparison
from safe_observation.opponents import leduc_opponent_suite

ROUNDS, EPISODES = 60, 200
SEEDS = list(range(42, 47))
DELTA, METHOD = 0.1, "empirical_bernstein"
EPS_SAFE, RHO_CAP, KAPPA, RNR_P = 0.0, 0.5, 0.2, 0.5
LEDGER_ON, LEDGER_OFF = 3.0, float("inf")
ARMS = ("public_robust", "point_response", "safe_active_decensoring")
OUT = Path("results/redundancy/ledger_ablation_leduc.json")


def _run(debt_max: float) -> dict:
    return run_showdown_comparison(
        leduc_opponent_suite(),
        rounds=ROUNDS,
        episodes_per_round=EPISODES,
        delta=DELTA,
        eps_safe=EPS_SAFE,
        method=METHOD,
        rho_cap=RHO_CAP,
        kappa=KAPPA,
        safety_debt_max=debt_max,
        rnr_p=RNR_P,
        seeds=SEEDS,
        out_dir=None,
        workers=10,
    )


def main() -> None:
    print("running Leduc suite: ledger ON (debt_max=3.0)...", flush=True)
    on = _run(LEDGER_ON)
    print("running Leduc suite: ledger OFF (debt_max=inf)...", flush=True)
    off = _run(LEDGER_OFF)

    floor = next(iter(on["opponents"].values()))["game_value"] - RHO_CAP
    rows: dict[str, dict] = {}
    for opp in on["opponents"]:
        rows[opp] = {}
        for arm in ARMS:
            mo = on["opponents"][opp]["methods"][arm]
            mf = off["opponents"][opp]["methods"][arm]
            rows[opp][arm] = {
                "gain_on": mo["exploitation_gain_mean"],
                "gain_off": mf["exploitation_gain_mean"],
                "minsafe_on": mo["min_safety_value"],
                "minsafe_off": mf["min_safety_value"],
                "spent_on": mo["budget_spent_mean"],
                "spent_off": mf["budget_spent_mean"],
                "violation_on": mo["safety_violation_max"],
                "violation_off": mf["safety_violation_max"],
            }

    out = {
        "game": "leduc",
        "floor": floor,
        "seeds": SEEDS,
        "rounds": ROUNDS,
        "episodes_per_round": EPISODES,
        "ledger_on_debt_max": LEDGER_ON,
        "arms": list(ARMS),
        "results": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}\n", flush=True)

    worst_violation = 0.0
    for opp in rows:
        for arm in ARMS:
            r = rows[opp][arm]
            worst_violation = max(
                worst_violation, r["violation_on"], r["violation_off"]
            )
    print(f"floor = {floor:.3f}")
    print(f"worst safety violation (on OR off, all arms) = {worst_violation:.2e}")
    print(
        "SAFETY HOLDS BOTH" if worst_violation <= RHO_CAP + 1e-9 else "*** BREACH ***",
        flush=True,
    )
    print(
        "\n=== safe_active_decensoring: ledger ON vs OFF (gain | minSafe | spend) ==="
    )
    print(
        f"{'opponent':18}{'gainOn':>8}{'gainOff':>8} | {'mSafeOn':>8}{'mSafeOff':>9} | {'spendOn':>8}{'spendOff':>9}"
    )
    for opp in rows:
        r = rows[opp]["safe_active_decensoring"]
        print(
            f"{opp:18}{r['gain_on']:>+8.3f}{r['gain_off']:>+8.3f} | "
            f"{r['minsafe_on']:>8.3f}{r['minsafe_off']:>9.3f} | "
            f"{r['spent_on']:>8.2f}{r['spent_off']:>9.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
