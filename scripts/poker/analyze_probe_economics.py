"""Analyze probe economics. See Experiments and supplementary Certification at the Unbucketed River."""

from collections import defaultdict
import json
from pathlib import Path
import statistics as st
import sys

GAME = sys.argv[1] if len(sys.argv) > 1 else "holdem_tr_b2"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000


sys.argv = ["run_safe_active_decensoring.py", GAME]
from scripts.poker import run_safe_active_decensoring as rsd

rsd._init(None, None)
W = rsd._W
payoff, sf0, sf1 = W["payoff"], W["sf0"], W["sf1"]
v_ref = W["v_ref"]

cells_path = Path(f"results/sad_deploy_{GAME}.cells.jsonl")
cells = [json.loads(line) for line in open(cells_path)]
by = defaultdict(list)
for c in cells:
    if c.get("lp_timeout"):
        continue
    by[(c["opponent"], c["mode"], c["reveal_budget"])].append(c)

rows = []
for name in rsd.LEAKS:
    if name == "tr_equilibrium":
        continue
    key, beh, safety = rsd._probe_task(("sad", name))
    if beh is None:
        beh = W["bp_behavior"]
    x_probe = sf0.realization_from_behavior(beh)
    y_star = W["y_stars"][name] if "y_stars" in W else None
    if y_star is None:
        y_star = list(
            rsd.Opponent(name=name, behavior=W["suite"][name], game=GAME).realization()
        )
    u_probe = payoff.bilinear(list(x_probe), y_star) - v_ref

    def mean_real(mode, opponent=name):
        """Compute mean real for the analyze probe economics workflow."""
        cs = by.get((opponent, mode, N), [])
        return st.mean(c["realized"] for c in cs) if cs else None

    u_cpub = mean_real("cpub") or st.mean(
        c["realized"] for c in by.get((name, "cpub", 0), [])
    )

    routed_reals = []
    cpub0 = by.get((name, "cpub", 0), []) or by.get((name, "cpub", N), [])
    cpub_cert = st.mean(c["certified"] for c in cpub0)
    for c in by.get((name, "sad", N), []):
        routed_reals.append(c["realized"] if c["certified"] >= cpub_cert else u_cpub)
    u_routed = st.mean(routed_reals)

    if u_routed <= u_cpub + 1e-9:
        regime, tstar, mult = "never", None, None
    elif u_probe >= u_cpub - 1e-9:
        regime, tstar, mult = "immediate", None, None
    else:
        tstar = N * (u_routed - u_probe) / (u_routed - u_cpub)
        regime, mult = "amortized", tstar / N
    rows.append(
        {
            "opponent": name,
            "N": N,
            "u_probe": round(u_probe, 6),
            "u_cpub": round(u_cpub, 6),
            "u_routed": round(u_routed, 6),
            "probe_cost_per_hand": round(u_cpub - u_probe, 6),
            "regime": regime,
            "crossover_T": round(tstar) if tstar else None,
            "payback_multiple": round(mult, 2) if mult else None,
        }
    )
    print(
        f"{name:<22} U_probe {u_probe:+.4f}  U_cpub {u_cpub:+.4f}  "
        f"U_routed {u_routed:+.4f}  regime={regime}"
        f"{f'  T*={round(tstar)} ({mult:.2f}x N)' if tstar else ''}"
    )

out = Path(f"results/probe_economics_{GAME}.json")
json.dump(
    {"game": GAME, "N": N, "v_ref": v_ref, "rows": rows}, open(out, "w"), indent=1
)
print(f"wrote {out}")
