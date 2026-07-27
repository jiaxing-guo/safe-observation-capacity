"""Render selective labels table. See Experiments and supplementary Additional Experiments."""

from __future__ import annotations

import json
from pathlib import Path

SRC = Path("results/sl_deploy.json")
OUT = Path("generated/tables/sl_deploy_table.tex")
OUT.parent.mkdir(parents=True, exist_ok=True)
RHO = 0.05
NS = [1_000, 10_000, 100_000]


def _g(summary, method, rho, n):
    """Compute g for the render selective labels table workflow."""
    return summary[f"{method}|{rho}|{n}"]


def main() -> None:
    """Run the command-line entry point."""
    d = json.loads(SRC.read_text())
    s = d["summary"]
    seeds = len(d["seeds"])
    eps, delta = d["eps"], d["delta"]
    wstar = d["W_star"]
    orc = _g(s, "oracle", RHO, 0)

    def row(method, n, label, bold=False):
        """Format one result row for the generated report."""
        cell = _g(s, method, RHO, n)
        cert, real = cell["certified"], cell["realized"]
        cov = cell["coverage"]
        viol = cell["violations"]
        nc = cell["n_cells"]
        b0, b1 = (r"\mathbf{", "}") if bold else ("", "")
        return (
            f"{label} & ${b0}{cert:.3f}{b1}$ & ${b0}{real:.3f}{b1}$ "
            f"& ${cov:.2f}$ & ${viol}/{nc}$ \\\\"
        )

    lines = [
        "\\begin{table}[!tb]",
        "\\centering\\small",
        "\\caption{Safe active de-censoring on the lending selective-labels instance "
        f"($\\rho={RHO}$, $\\delta={delta}$, $\\epsilon={eps}$; mean over {seeds} matched "
        "seeds; robustly-safe baseline value $W^\\star"
        f"={wstar:.3f}$).  The \\emph{{passive}} arm trusts the historical (acted-on) labels and "
        "is MNAR-biased: its certificate's coverage of the true repay rates \\emph{collapses} "
        "from $0.50$ to $0.00$ as $N$ grows (Theorem~\\ref{thm:passive_mnar} analogue).  The "
        "\\emph{active} arm spends the safety slack to draw unbiased labels on the censored "
        "strata; its coverage holds near $1$ and its realized value climbs toward the "
        "\\emph{oracle} (which knows the true rates).  \\emph{Cover.}\\ is the fraction of seeds "
        "whose confidence set contains the truth; \\emph{Viol.}\\ is the floor-violation rate "
        "(Theorem~\\ref{thm:safety}).  Certified value also grows with the budget "
        f"($\\rho=0,0.01,{RHO}$: active $"
        f"{_g(s, 'active', 0.0, 100000)['certified']:.3f}\\!\\to\\!"
        f"{_g(s, 'active', 0.01, 100000)['certified']:.3f}\\!\\to\\!"
        f"{_g(s, 'active', RHO, 100000)['certified']:.3f}$ at $N=10^5$).}}",
        "\\label{tab:sl_deploy}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}llrrrr@{}}",
        "\\toprule",
        "Method & $N$ & Cert. & Real. & Cover. & Viol. \\\\",
        "\\midrule",
    ]
    for n in NS:
        lab = "Passive (MAR)" if n == NS[0] else ""
        lines.append(row("passive", n, f"{lab} & $10^{{{len(str(n)) - 1}}}$"))
    lines.append("\\midrule")
    for n in NS:
        lab = "Active de-censoring" if n == NS[0] else ""
        bold = n == NS[-1]
        lines.append(row("active", n, f"{lab} & $10^{{{len(str(n)) - 1}}}$", bold=bold))
    lines.append("\\midrule")
    cert, real = orc["certified"], orc["realized"]
    lines.append(
        f"Oracle (true rates) & --- & ${cert:.3f}$ & ${real:.3f}$ & "
        f"${orc['coverage']:.2f}$ & ${orc['violations']}/{orc['n_cells']}$ \\\\"
    )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    print(
        "  passive coverage by N: "
        + ", ".join(f"{_g(s, 'passive', RHO, n)['coverage']:.2f}" for n in NS)
    )
    print(
        "  active  coverage by N: "
        + ", ".join(f"{_g(s, 'active', RHO, n)['coverage']:.2f}" for n in NS)
    )
    print(
        "  active realized by N : "
        + ", ".join(f"{_g(s, 'active', RHO, n)['realized']:.3f}" for n in NS)
        + f"  (oracle {orc['realized']:.3f})"
    )
    print(f"  floor violations total: {d['total_violations']}")


if __name__ == "__main__":
    main()
