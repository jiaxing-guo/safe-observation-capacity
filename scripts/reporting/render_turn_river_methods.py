"""Render turn river methods. See Experiments and supplementary Certification at the Unbucketed River."""

import json
import os
from pathlib import Path

IN_JSON = Path(
    os.environ.get(
        "TR_METHOD_TABLE_IN", "results/turn_river_method_table_holdem_tr_b4_25s.json"
    )
)
OUT_TEX = Path(
    os.environ.get(
        "TR_METHOD_TABLE_TEX", "generated/tables/turn_river_method_table.tex"
    )
)

OPPONENT_LABELS = {
    "tr_equilibrium": "Equilibrium",
    "tr_river_overfold_uniform": "Late over-fold, uniform",
    "tr_river_overfold_strong": "Late over-fold, strong",
    "tr_turn_overfold": "Early over-fold",
}
OPPONENT_ORDER = [
    "tr_equilibrium",
    "tr_river_overfold_uniform",
    "tr_river_overfold_strong",
    "tr_turn_overfold",
]
RHO_ORDER = [0.1, 0.5]


def _stats(aggregate: list[dict], opponent: str, rho: float, metric: str) -> dict:
    """Compute summary statistics across independent replicates."""
    for row in aggregate:
        if row["opponent"] == opponent and abs(float(row["rho"]) - rho) < 1e-9:
            return row["metrics"][metric]
    raise KeyError((opponent, rho, metric))


def _fmt_pm(stats: dict) -> str:
    """Compute fmt pm for the render turn river methods workflow."""
    mean = float(stats["mean"])
    ci = float(stats["ci95"])
    return f"${mean:+.3f}\\pm{ci:.3f}$"


def _fmt_plain(value: float) -> str:
    """Compute fmt plain for the render turn river methods workflow."""
    if abs(value) < 5e-6:
        value = 0.0
    return f"${value:+.3f}$"


def _row(aggregate: list[dict], opponent: str, rho: float) -> str:
    """Format one result row for the generated report."""
    label = OPPONENT_LABELS[opponent]
    oracle = _fmt_pm(_stats(aggregate, opponent, rho, "oracle"))
    core = _fmt_pm(_stats(aggregate, opponent, rho, "core"))
    em = _fmt_pm(_stats(aggregate, opponent, rho, "em"))
    pub = _fmt_pm(_stats(aggregate, opponent, rho, "pub"))
    gap = _fmt_pm(_stats(aggregate, opponent, rho, "core_minus_em"))
    cert = _fmt_pm(_stats(aggregate, opponent, rho, "core_cert_exploit"))
    safety = _fmt_plain(float(_stats(aggregate, opponent, rho, "safety_margin")["min"]))
    return (
        f"{label} & {rho:.1f} & {oracle} & {core} & {em} & {pub} & "
        f"{gap} & {cert} & {safety} \\\\"
    )


def main() -> None:
    """Run the command-line entry point."""
    data = json.load(IN_JSON.open())
    aggregate = data["aggregate"]
    body = "\n".join(
        _row(aggregate, opponent, rho)
        for opponent in OPPONENT_ORDER
        for rho in RHO_ORDER
    )
    tex = f"""\\begin{{table*}}[t]
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{2.2pt}}
\\begin{{tabular}}{{llrrrrrrr}}
\\toprule
Opponent & $\\rho$ & Oracle & Robust $\\Cpub$ & \\cem{{}} & Public point & Robust--EM & Cert. & Safety \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{Turn--river opponent--method table over 25 matched seeds.  Entries are
mean $\\pm$ 95\\% confidence interval.  Robust--EM is the realized gain difference
against the true opponent; Cert. is the robust worst-case gain over the public
observation fiber; Safety is the worst verified margin above the declared floor.}}
\\label{{tab:tr_method_shape}}
\\end{{table*}}
"""
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex)
    print(f"wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
