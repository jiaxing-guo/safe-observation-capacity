"""Check value fiber. See supplementary Reproducibility for its role in the release workflow."""

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


def _hist(label: str) -> str:
    """Compute hist for the check value fiber workflow."""
    return label.split("|", 1)[1] if "|" in label else ""


def _decompose(value: dict[str, float]) -> dict[str, float]:
    """Compute decompose for the check value fiber workflow."""
    by_state: dict[str, list[float]] = {}
    for label, v in value.items():
        by_state.setdefault(_hist(label), []).append(v)
    allv = list(value.values())
    n = len(allv)
    if n < 2:
        return {}
    mean = sum(allv) / n
    total_var = sum((v - mean) ** 2 for v in allv) / n

    between = 0.0
    within = 0.0
    for vs in by_state.values():
        ns = len(vs)
        sm = sum(vs) / ns
        between += ns * (sm - mean) ** 2
        within += sum((v - sm) ** 2 for v in vs)
    between_var = between / n
    within_var = within / n

    multi_val = sum(sum(vs) for vs in by_state.values() if len(vs) > 1)
    tot_val = sum(allv)
    return {
        "n_targets": n,
        "n_states": len(by_state),
        "total_var": total_var,
        "between_frac": between_var / total_var if total_var > 0 else float("nan"),
        "within_frac": within_var / total_var if total_var > 0 else float("nan"),
        "multi_state_value_frac": multi_val / tot_val if tot_val > 0 else float("nan"),
    }


def _dpub_const_within(dpub: dict[str, float], value: dict[str, float]) -> float:
    """Compute dpub const within for the check value fiber workflow."""
    by_state: dict[str, list[float]] = {}
    for label in value:
        by_state.setdefault(_hist(label), []).append(dpub.get(label, 0.0))
    spreads = [max(s) - min(s) for s in by_state.values() if len(s) > 1]
    return max(spreads) if spreads else 0.0


def main() -> None:
    """Run the command-line entry point."""
    D._init()
    print(f"# Public-fiber value-identifiability smoke on {GAME}\n")
    print(
        f"{'family':<22}{'n_tgt':>6}{'n_state':>8}{'between':>9}{'within':>8}"
        f"{'multi%':>8}{'Dpub_spread':>12}"
    )
    for fam in DEV:
        beh = D._W["suite"][fam]
        y_star = list(Opponent(name=fam, behavior=beh, game=GAME).realization())
        value = D._leak_weights(y_star)
        dpub = D._public_anomaly_weights(y_star)
        dec = _decompose(value)
        if not dec:
            print(f"{fam:<22}  (insufficient targets)")
            continue
        spread = _dpub_const_within(dpub, value)
        print(
            f"{fam:<22}{dec['n_targets']:>6d}{dec['n_states']:>8d}"
            f"{dec['between_frac']:>9.3f}{dec['within_frac']:>8.3f}"
            f"{dec['multi_state_value_frac']:>8.2f}{spread:>12.2e}"
        )
    print(
        "\nbetween = value variance D_pub CAN rank (across public states)\n"
        "within  = value variance D_pub is BLIND to (across holes in a state)\n"
        "multi%  = leak value in public states with >1 value target\n"
        "Dpub_spread = max within-state spread of D_pub (≈0 ⇒ D_pub is a public-"
        "state function)"
    )


if __name__ == "__main__":
    main()
