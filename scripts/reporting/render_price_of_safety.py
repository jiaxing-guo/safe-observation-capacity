"""Render price of safety. See Experiments and supplementary Additional Experiments."""

from __future__ import annotations

import json
from pathlib import Path

OMAX = 1.0 / 3.0


def _kuhn_curves():
    """Compute Kuhn curves for the render price of safety workflow."""
    d = json.load(open("results/spike_tradeoff_law_kuhn.json"))

    pick = {"2:b": "low floor", "0:b": "mid floor", "1:b": "high floor"}
    out = []
    for key, label in pick.items():
        pts = [(p["rho"], p["kappa"]) for p in d["curves"][key]]
        k0 = pts[0][1]

        mu = (pts[1][1] - k0) / (pts[1][0] - pts[0][0])

        rho_sat = (OMAX - k0) / mu
        out.append(
            {
                "key": key,
                "label": label,
                "pts": pts,
                "k0": k0,
                "mu": mu,
                "rho_sat": rho_sat,
            }
        )
    return out


def _bandit_wall():
    """Compute bandit wall for the render price of safety workflow."""
    d = json.load(open("results/probe_bandit_sao.json"))
    w = d["pure_wall"]["curve"]
    sigma, delta = d["sigma"], d["delta"]
    pts = []
    for c in w:
        if 0 < c["rho"] <= 1.0 and c["kappa"] > 1e-9:
            n = sigma**2 / (delta**2 * c["kappa"])
            pts.append((c["rho"], n))
    const = sigma**2 * d["pure_wall"]["L"] / delta**2
    return pts, const


def _holdem_ncert():
    """Compute holdem ncert for the render price of safety workflow."""
    d = json.load(open("results/kappa_rho_sweep_holdem_tr_b2.json"))
    pr = d["per_rho"]
    pts = sorted((float(r), pr[r]["median_N_cert"]) for r in pr)
    return pts


def _leduc_multiface():
    """Compute Leduc multiface for the render price of safety workflow."""
    d = json.load(open("results/spike_tradeoff_law_leduc.json"))
    pts = [(p["rho"], p["kappa"]) for p in d["curves"]["K|-|crr|"]]
    k0 = pts[0][1]
    mu = (pts[1][1] - k0) / (pts[1][0] - pts[0][0])
    wmax = max(k for _, k in pts)
    return pts, k0, mu, wmax


def _coords(pts, fx=lambda x: x, fy=lambda y: y):
    """Serialize the selected observations as plotting coordinates."""
    return " ".join(f"({fx(x):.5g},{fy(y):.5g})" for x, y in pts)


def main() -> None:
    """Run the command-line entry point."""
    wall, const = _bandit_wall()
    holdem = _holdem_ncert()
    leduc_pts, lk0, lmu, lwmax = _leduc_multiface()

    a_lines = []

    fpts = [(r, k) for r, k in leduc_pts if r <= 0.46]
    a_lines.append(
        f"\\addplot[blue, thick, mark=*, mark size=1.2pt] coordinates {{{_coords(fpts)}}};"
    )

    a_lines.append(
        "\\node[blue, font=\\scriptsize, anchor=north] at (axis cs:0.225,0.205) "
        "{$\\kappa_\\rho(I)$};"
    )

    rho_hit = (lwmax - lk0) / lmu
    a_lines.append(
        f"\\addplot[black, densely dashed, forget plot] coordinates "
        f"{{(0,{lk0:.4f}) ({rho_hit:.4f},{lwmax:.4f})}} "
        f"node[pos=0.55, sloped, above=1pt, font=\\scriptsize, text=black!80] "
        f"{{$\\kappa_0{{+}}\\mu_I\\rho$}};"
    )

    a_lines.append(
        f"\\addplot[gray, densely dotted, forget plot] coordinates "
        f"{{(0,{lwmax:.4f}) (0.46,{lwmax:.4f})}};"
    )
    a_lines.append(
        f"\\node[gray, font=\\scriptsize, anchor=south east] at (axis cs:0.46,{lwmax:.4f}) "
        f"{{$\\omega_{{\\max}}$}};"
    )

    b_lines = []
    b_lines.append(
        f"\\addplot[black!75, very thick, mark=o, mark size=1.3pt] "
        f"coordinates {{{_coords(wall)}}};"
    )
    b_lines.append(
        "\\node[black!75, font=\\scriptsize, anchor=north] at (axis cs:0.055,250) "
        "{bandit (wall)};"
    )
    b_lines.append(
        f"\\addplot[red!70!black, thick, mark=square*, mark size=1.2pt] "
        f"coordinates {{{_coords(holdem)}}};"
    )
    b_lines.append(
        "\\node[red!70!black, font=\\scriptsize, anchor=north] at (axis cs:0.31,1.05e5) "
        "{hold'em deep turn};"
    )

    ref_x = 0.014
    ref_y = const / ref_x * 1.12
    tex = (
        TEMPLATE.replace("@PANEL_A@", "\n".join(a_lines))
        .replace("@PANEL_B@", "\n".join(b_lines))
        .replace("@CONST@", f"{const:.0f}")
        .replace("@OMAX@", f"{OMAX:.4f}")
        .replace("@REFX@", f"{ref_x:.5g}")
        .replace("@REFY@", f"{ref_y:.5g}")
    )
    path = "generated/figures/fig_price_of_safety.tex"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(tex)
    print(f"wrote {path}")
    print(f"  bandit pure-wall N_cert*rho = {const:.1f} (constant => N ~ 1/rho)")
    print(
        f"  Leduc K|-|crr|: kappa_0={lk0:.4f}  origin mu_I={lmu:.3f}  omega_max={lwmax:.4f}"
        f"  (origin line hits omega_max at rho={(lwmax - lk0) / lmu:.3f}; actual frontier far below)"
    )


TEMPLATE = r"""\documentclass[border=3pt]{standalone}
\usepackage{amsmath}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{calc}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=2 by 1, horizontal sep=1.9cm},
  width=6.8cm, height=5.3cm,
  tick label style={font=\footnotesize}, label style={font=\small},
  title style={font=\small, yshift=-1pt},
  grid=both, grid style={gray!16},
]
\nextgroupplot[
  xlabel={safety budget $\rho$},
  ylabel={safe reveal capacity $\kappa_\rho(I)$},
  xmin=0, xmax=0.46, ymin=0, ymax=0.36,
  xtick={0,0.1,0.2,0.3,0.4},
  ytick={0,0.1,0.2,0.3,0.3333},
  yticklabels={$0$,$0.1$,$0.2$,$0.3$,$\omega_{\max}$},
]
@PANEL_A@
\nextgroupplot[
  xlabel={safety budget $\rho$},
  ylabel={certification cost $N_{\mathrm{cert}}$},
  xmode=log, log basis x=10, ymode=log, log basis y=10,
  xmin=0.008, xmax=1.2,
]
\addplot[gray!75, densely dashed, domain=0.01:1, samples=2, forget plot]
  {@CONST@/x};
@PANEL_B@
\node[gray!75, font=\scriptsize, anchor=south west, inner sep=0.5pt]
    at (axis cs:@REFX@,@REFY@) {$\propto 1/\rho$};
\end{groupplot}
\node[font=\small, anchor=north] at ($(group c1r1.south)+(0,-30pt)$)
  {(a) frontier $\le$ origin dual line};
\node[font=\small, anchor=north] at ($(group c2r1.south)+(0,-30pt)$)
  {(b) certification cost $N_{\mathrm{cert}}$};
\end{tikzpicture}
\end{document}
"""

if __name__ == "__main__":
    main()
