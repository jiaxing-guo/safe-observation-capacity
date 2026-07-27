"""Render identification coverage. See Experiments and supplementary Additional Experiments."""

import json
from pathlib import Path
import sys

GAME = sys.argv[1] if len(sys.argv) > 1 else "holdem_tr_b2"

FAM = {
    "river_overfold_w80": ("River over-fold", "blue", "*", "RIVER"),
    "turn_overfold_w70": ("Turn over-fold", "red!70!black", "square*", "TURN"),
    "revealed_call_strong": ("Revealed call", "green!55!black", "triangle*", "REV"),
}

data = json.loads(open(f"results/cid_coverage_{GAME}.json").read())


def coords(leak: str, key: str) -> str:
    """Serialize the selected observations as plotting coordinates."""
    by_n = data[leak]["by_N"]
    return "".join(
        f"({int(n)},{by_n[n][key]:.6g})" for n in sorted(by_n, key=lambda s: int(s))
    )


repl = {}
for leak, (_label, _color, _mark, tag) in FAM.items():
    repl[f"__{tag}_ACT__"] = coords(leak, "coverage")
    repl[f"__{tag}_PASS__"] = coords(leak, "passive_coverage")
    repl[f"__{tag}_W__"] = coords(leak, "mean_width")

TEMPLATE = r"""\documentclass[border=3pt]{standalone}
\usepackage{amsmath}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{calc}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=2 by 1, horizontal sep=1.7cm},
  width=6.7cm, height=5.2cm,
  xmode=log, log basis x=10,
  xtick={1000,10000,100000,1000000},
  xticklabels={$10^3$,$10^4$,$10^5$,$10^6$},
  xlabel={reveal budget $N$},
  tick label style={font=\footnotesize}, label style={font=\small},
  title style={font=\small, yshift=-1pt},
  grid=both, grid style={gray!16},
]
\nextgroupplot[
  ylabel={fraction covering $y^\star$},
  ymin=0.6, ymax=1.04, ytick={0.6,0.7,0.8,0.9,1.0}]
\draw[gray, densely dotted] (axis cs:800,0.9) -- (axis cs:1.35e6,0.9);
\node[gray, font=\scriptsize, anchor=south east] at (axis cs:1.35e6,0.902)
  {nominal $1-\delta$};
\addplot[blue, dashed, mark=*, mark size=1.1pt] coordinates {__RIVER_PASS__};
\addplot[red!70!black, dashed, mark=square*, mark size=1.1pt] coordinates {__TURN_PASS__};
\addplot[green!55!black, dashed, mark=triangle*, mark size=1.3pt] coordinates {__REV_PASS__};
\addplot[blue, thick, mark=*, mark size=1.1pt] coordinates {__RIVER_ACT__};
\addplot[red!70!black, thick, mark=square*, mark size=1.1pt] coordinates {__TURN_ACT__};
\addplot[green!55!black, thick, mark=triangle*, mark size=1.3pt] coordinates {__REV_ACT__};
\node[anchor=south west, font=\scriptsize] at (axis cs:1000,1.006)
  {active $C_{\mathrm{id}}$};
\node[anchor=north east, font=\scriptsize] at (axis cs:1.1e6,0.72) {passive box};
\nextgroupplot[
  ylabel={mean width $\eta$}, ymode=log, log basis y=10,
  legend to name=famlegend, legend columns=3,
  legend style={font=\footnotesize, draw=none, column sep=1ex}]
\addplot[black, densely dashed, domain=1e4:1e6, samples=2, forget plot]
  {0.013*(x/1e4)^(-0.5)};
\node[font=\scriptsize, anchor=south west] at (axis cs:1.15e4,0.0145)
  {$\propto N^{-1/2}$};
\addplot[blue, thick, mark=*, mark size=1.1pt] coordinates {__RIVER_W__};
\addlegendentry{River over-fold}
\addplot[red!70!black, thick, mark=square*, mark size=1.1pt] coordinates {__TURN_W__};
\addlegendentry{Turn over-fold}
\addplot[green!55!black, thick, mark=triangle*, mark size=1.3pt] coordinates {__REV_W__};
\addlegendentry{Revealed call}
\end{groupplot}
\node[anchor=south] at ($(group c1r1.north)!0.5!(group c2r1.north)+(0,4pt)$)
  {\pgfplotslegendfromname{famlegend}};
\node[font=\small, anchor=north] at ($(group c1r1.south)+(0,-30pt)$)
  {(a) coverage of $y^\star$};
\node[font=\small, anchor=north] at ($(group c2r1.south)+(0,-30pt)$)
  {(b) active $C_{\mathrm{id}}$ width vs.\ budget};
\end{tikzpicture}
\end{document}
"""

tex = TEMPLATE.replace("GAME", GAME)
for k, v in repl.items():
    tex = tex.replace(k, v)

out = "generated/figures/fig_cid_coverage.tex"
Path(out).parent.mkdir(parents=True, exist_ok=True)
open(out, "w").write(tex)
print(f"wrote {out}")
