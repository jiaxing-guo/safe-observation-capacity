"""Render deployment rate. See Experiments and supplementary Additional Experiments."""

from __future__ import annotations

import json
from pathlib import Path
import statistics as st
import sys

DEV = ["river_overfold_w80", "turn_overfold_w70", "revealed_call_strong"]
SHORT = {
    "river_overfold_w80": "river over-fold",
    "turn_overfold_w70": "deep turn over-fold",
    "revealed_call_strong": "revealed call",
}


def _load_rows():
    """Load rows for the render deployment rate workflow."""
    rows = []
    base = "results/sad_deploy_holdem_tr_b2.json"
    with open(base) as fh:
        rows += json.load(fh)["rows"]

    try:
        with open("results/sad_deploy_holdem_tr_b2_n1e6.json") as fh:
            rows += json.load(fh)["rows"]
    except FileNotFoundError:
        pass
    return rows


def main() -> None:
    """Run the command-line entry point."""
    rows = _load_rows()
    oracle = {r["opponent"]: r["certified"] for r in rows if r["mode"] == "oracle"}
    Ns = sorted({r["episodes"] for r in rows if r["mode"] == "sad"})
    print(f"# budgets present: {Ns}", file=sys.stderr)

    lines = []
    for opp in DEV:
        coords = []
        for n in Ns:
            cs = [
                r["certified"]
                for r in rows
                if r["opponent"] == opp and r["mode"] == "sad" and r["episodes"] == n
            ]
            if cs:
                coords.append((n, st.mean(cs)))
        body = " ".join(f"({n},{v:.4f})" for n, v in coords)
        lines.append(f"\\addplot+[mark=*] coordinates {{{body}}};")
        lines.append(
            f"\\addplot[dashed,gray] coordinates "
            f"{{({Ns[0]},{oracle[opp]:.4f}) ({Ns[-1]},{oracle[opp]:.4f})}};"
        )
    out = "\n".join(lines)
    path = "generated/figures/fig_deploy_rate_data.tex"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(out + "\n")
    print(out)
    print(f"\n# wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
