"""Render turn river reach. See Experiments and supplementary Certification at the Unbucketed River."""

import json
import os
from pathlib import Path

IN_JSON = Path(
    os.environ.get(
        "TR_REACH_TABLE_IN",
        "results/turn_river_online_reach_after_method_table_smoke.json",
    )
)
OUT_TEX = Path(
    os.environ.get("TR_REACH_TABLE_TEX", "generated/tables/turn_river_reach_table.tex")
)
RHO_CAP = float(os.environ.get("TR_REACH_RHO_CAP", "0.5"))

ARM_LABELS = {
    "core": "Robust $\\Cpub$",
    "obs": "Robust $C_{\\mathrm{obs}}$",
    "em": "\\cem{}",
    "obs_seed": "EM$\\to C_{\\mathrm{obs}}$",
}
ARM_ORDER = ["core", "obs", "em", "obs_seed"]


def _fmt(value: float | None, digits: int = 3) -> str:
    """Compute fmt for the render turn river reach workflow."""
    if value is None:
        return "n/a"
    if abs(value) < 5e-6:
        value = 0.0
    return f"${value:+.{digits}f}$"


def _fmt_reach(value: float | None) -> str:
    """Compute fmt reach for the render turn river reach workflow."""
    if value is None:
        return "n/a"
    return f"${value:.2f}$"


def _first_v_ref(results: list[dict]) -> float | None:
    """Compute first v ref for the render turn river reach workflow."""
    for result in results:
        if not result.get("ok"):
            continue
        out_path = Path(result["out"])
        if not out_path.exists():
            continue
        detail = json.load(out_path.open())
        return float(detail["v_ref"])
    return None


def main() -> None:
    """Run the command-line entry point."""
    data = json.load(IN_JSON.open())
    aggregate = data["aggregate"]
    v_ref = _first_v_ref(data["results"])
    floor = (v_ref - RHO_CAP) if v_ref is not None else None
    rows = []
    for arm in ARM_ORDER:
        item = aggregate[arm]
        safe_margin = None
        if floor is not None and item["worst_min_safety"] is not None:
            safe_margin = float(item["worst_min_safety"]) - floor
        rows.append(
            f"{ARM_LABELS[arm]} & {_fmt(item['last5_realized_mean'])} & "
            f"{_fmt(item['last5_cert_mean'])} & "
            f"{_fmt_reach(item['final_exploit_reach_mean'])} & "
            f"{_fmt(safe_margin)} \\\\"
        )
    body = "\n".join(rows)
    tex = f"""\\begin{{table}}[t]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
Agent & Real. & Cert. & Reach & Safe \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{Turn--river online reach-enrichment table.  Real. and Cert. are
final-window averages; Reach is final accumulated reach on oracle-relevant low-reach
information sets; Safe is the worst verified value minus the floor.}}
\\label{{tab:tr_reach_shape}}
\\end{{table}}
"""
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex)
    print(f"wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
