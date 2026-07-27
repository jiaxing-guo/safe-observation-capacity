"""Inspect whether candidate poker lines satisfy reveal certification."""

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
from scripts.poker import run_safe_active_decensoring as D


def main() -> None:
    """Run the command-line entry point."""
    D._init()
    sf1, y_eq = D._W["sf1"], D._W["y_eq"]
    by_hist: dict[str, dict] = {}
    for info in sf1.info_sets:
        if not D._is_river(info.label):
            continue
        h = info.label.split("|", 1)[1]
        d = by_hist.setdefault(h, {"n": 0, "reach": 0.0})
        d["n"] += 1
        d["reach"] += y_eq[info.parent_seq]
    rows = sorted(by_hist.items(), key=lambda kv: -kv[1]["reach"])
    print(f"# {len(rows)} river public lines (coarse)\n")
    print(f"{'hist':<28}{'n_targets':>10}{'eq_reach':>12}")
    for h, d in rows[:30]:
        print(f"{h:<28}{d['n']:>10d}{d['reach']:>12.4f}")


if __name__ == "__main__":
    main()
