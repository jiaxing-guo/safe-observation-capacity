"""Render full river report. See Experiments and supplementary Certification at the Unbucketed River."""

import json
from pathlib import Path
import re
import statistics as st

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
FIG = ROOT / "generated" / "figures"
TAB = ROOT / "generated" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

ed = json.load(open(R / "ed_full_river_merged.json"))
res = json.load(open(R / "full_river_residual_reach_invariant.json"))
ee = json.load(open(R / "e_e_solver_table.json"))
drift = json.load(open(R / "drift_full_river_merged.json"))
zoo = json.load(open(R / "zoo_full_river.json"))

rows = {(r["leak"], r["n_hands"], r["arm"]): r for r in ed["rows"]}
NS = [100_000, 1_000_000, 10_000_000]
V_SAFE = res["v_safe"]
V_REF = res["v_ref"]
GRP = next(d for d in res["designs"] if d["design"] == "grouped_k4")
PUB = next(d for d in res["designs"] if d["design"] == "public_only")
PC = next(d for d in res["designs"] if d["design"] == "percombo")
ASYM = GRP["r_m"]

viol = sum(r["floor_violations"] for r in ed["rows"])
assert viol == 0, f"floor violations in merged results: {viol}"


NCELL = sum(r["seeds"] for r in ed["rows"] if r["arm"] == "routed")


def sub(template, mapping):
    """Substitute named values into a report template."""
    for k, v in mapping.items():
        template = template.replace("@" + k + "@", v)
    leftover = re.findall(r"@[A-Z][A-Z0-9]*@", template)
    assert not leftover, f"unsubstituted tokens: {leftover}"
    return template


def coords(leak, arm):
    """Serialize the selected observations as plotting coordinates."""
    out = []
    for n in NS:
        r = rows[(leak, n, arm)]
        out.append(f"({n},{r['cert_mean']:.4f}) +- (0,{r['cert_se']:.4f})")
    return " ".join(out)


fig = sub(
    r"""\documentclass[border=3pt]{standalone}
\usepackage{amsmath}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
  width=7.8cm, height=4.3cm,
  xmode=log, log basis x=10,
  xtick={100000,1000000,10000000},
  xticklabels={$10^5$,$10^6$,$10^7$},
  xlabel={probe budget $N$ (hands)},
  ylabel={certified value},
  ymin=-0.13, ymax=0.9,
  tick label style={font=\footnotesize},
  label style={font=\small},
  legend style={font=\scriptsize, at={(0.5,1.03)}, anchor=south, draw=none,
    fill=none, legend columns=3, column sep=0.5em, inner sep=1pt},
  legend cell align=left,
]
\addplot[gray, dotted, thick, domain=100000:10000000] {@VSAFE@};
\addlegendentry{safe $V{=}@VSAFE3@$}
\addplot[gray, dashed, domain=100000:10000000] {@ASYM@};
\addlegendentry{fiber $R_M{=}@ASYM3@$}
\addplot[blue, thick, mark=*, mark size=1.6pt, error bars/.cd, y dir=both, y explicit]
  coordinates {@GRPC@};
\addlegendentry{$C^{\mathrm{id}}$, bucketed pins}
\addplot[blue!55, thick, mark=triangle*, mark size=1.9pt, dash dot, error bars/.cd, y dir=both, y explicit]
  coordinates {@PERCOMBO@};
\addlegendentry{$C^{\mathrm{id}}$, per-combo pins}
\addplot[teal!80!black, thick, mark=diamond*, mark size=2.0pt, densely dashed, error bars/.cd, y dir=both, y explicit]
  coordinates {@PASSIVE@};
\addlegendentry{passive (no probes)}
\addplot[red, thick, mark=square*, mark size=1.5pt, error bars/.cd, y dir=both, y explicit]
  coordinates {@PUBC@};
\addlegendentry{public $C^{\mathrm{pub}}$ (flat, SE $0$)}
\end{axis}
\end{tikzpicture}
\end{document}
""",
    {
        "VSAFE3": f"{V_SAFE:.3f}",
        "VSAFE": f"{V_SAFE:.4f}",
        "ASYM3": f"{ASYM:.3f}",
        "ASYM": f"{ASYM:.4f}",
        "GRPC": coords("reach_invariant", "C_id+grp"),
        "PERCOMBO": coords("reach_invariant", "C_id"),
        "PASSIVE": coords("reach_invariant", "C_id_passive"),
        "PUBC": coords("reach_invariant", "C_pub"),
    },
)
(FIG / "fig_full_river_twin_ladder.tex").write_text(fig)


def se_chip(v):
    """Compute se chip for the render full river report workflow."""
    return ("\\pm" + f"{v:.3f}".lstrip("0")) if v >= 0.0005 else "\\pm.000"


def cell(leak, n, arm="routed"):
    """Run one independently reproducible experiment cell."""
    return f"${rows[(leak, n, arm)]['cert_mean']:.3f}$"


def realized(leak):
    """Evaluate a realization plan against the selected opponent."""
    return f"${rows[(leak, 10_000_000, 'routed')]['real_mean']:.3f}$"


FAM = [
    ("control", "Equilibrium (control)"),
    ("overfold", "River over-fold"),
    ("revealed_call", "Revealed call"),
    ("sampled", "Sampled (hash-seeded)"),
    ("reach_invariant", r"\textbf{Public twin}"),
]
BODY_NAME = {
    "control": "Control (equil.)",
    "overfold": "River over-fold",
    "revealed_call": "Revealed call",
    "sampled": "Sampled",
    "reach_invariant": r"\textbf{Public twin}",
}
body_rows = "\n".join(
    f"{BODY_NAME[k]} & {cell(k, NS[0])} & {cell(k, NS[1])} & {cell(k, NS[2])} & {realized(k)} \\\\"
    for k, _ in FAM
)
body = sub(
    r"""\begin{table}[t]
\centering\footnotesize
\caption{The public channel never certifies the twin; the reveal channel
certifies $96\%$ of it.  Deployment on the fixed-board unbucketed river
subgame ($1081$ combos, $\rok=0.5$, $\vref=@VREF3@$; routed arm, mean over
$10$ seeds, seed SE $\le 0.004$, per-arm SEs in
App.~\ref{app:fulldeck_grid}; \emph{Real.}\ is realized value at
$N{=}10^7$; floors verified exactly, $0/@NCELL@$).  The \emph{public twin}
matches the equilibrium base in every public aggregate yet carries
$V=@VSAFE3@$ of safe value.}
\label{tab:ed_full_river}
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Opponent & $10^5$ & $10^6$ & $10^7$ & Real. \\
\midrule
@ROWS@
\bottomrule
\end{tabular}
\end{table}
""",
    {
        "ROWS": body_rows,
        "VREF3": f"{V_REF:.3f}",
        "VSAFE3": f"{V_SAFE:.3f}",
        "NCELL": str(NCELL),
    },
)
(TAB / "ed_full_river_deploy_table.tex").write_text(body)


pf_lines = []
for k, name in FAM:
    for i, n in enumerate(NS):
        arms = [rows[(k, n, a)] for a in ["C_pub", "C_id_passive", "C_id", "C_id+grp"]]
        first = name if i == 0 else ""
        pf_lines.append(
            f"{first} & $10^{{{len(str(n)) - 1}}}$ & "
            + " & ".join(
                f"${a['cert_mean']:.3f}\\,{{\\scriptsize\\pm {a['cert_se']:.3f}}}$"
                for a in arms
            )
            + f" & ${rows[(k, n, 'routed')]['real_mean']:.3f}$ \\\\"
        )
    pf_lines.append(r"\addlinespace[1.5pt]")
perfam = sub(
    r"""\begin{table*}[t]
\centering\footnotesize
\setlength{\tabcolsep}{5pt}
\caption{Full E-D grid at the unbucketed river: certified value of each
evidence arm (mean $\pm$ SE, $10$ seeds) and realized value of the routed
arm. Floors are verified exactly in every cell ($0$ violations in $@NCELL@$).
On the public twin, $\Cpub$ and per-combo $\Cid$ are numerically identical
until the pins gain statistical weight; bucketed pins certify from
$N{=}10^5$.}
\label{tab:ed_full_river_pf}
\begin{tabular}{@{}llrrrrr@{}}
\toprule
Opponent & $N$ & $\Cpub$ & passive & $\Cid$ per-combo & $\Cid$ bucketed & Real. \\
\midrule
@ROWS@
\bottomrule
\end{tabular}
\end{table*}
""",
    {"ROWS": "\n".join(pf_lines[:-1]), "NCELL": str(NCELL)},
)
(TAB / "ed_full_river_perfamily_table.tex").write_text(perfam)


def res_row(label, d):
    """Compute res row for the render full river report workflow."""
    if d.get("bracketed_by_monotonicity"):
        return (
            label
            + r" & --- & $\ge "
            + f"{d['r_m_lower']:.4f}"
            + r"$ & $\le "
            + f"{d['delta_minus_upper']:.4f}"
            + r"$ \\"
        )
    return (
        label
        + f" & ${d['rows']}$ & ${d['r_m']:.4f}$ & ${d['delta_minus']:.4f}$ "
        + r"\\"
    )


resid = sub(
    r"""\begin{table}[t]
\centering\small
\caption{Residual width of each evidence design against the public twin at
the population limit ($V=@VSAFE@$, $\vref=@VREF@$): robust-over-fiber value
$R_M$ (Thm.~\ref{thm:fiber_minimax}) and one-sided width
$\Delta^-_M=V-R_M$. Public rows identify nothing ($R_M=\vref$); bucketed
reveal pins recover all but $4\%$ of the safe value
(Thm.~\ref{thm:residual_recovery}).}
\label{tab:full_river_residual}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrr@{}}
\toprule
Evidence design & rows & $R_M$ & $\Delta^-_M$ \\
\midrule
@ROWS@
\bottomrule
\end{tabular}
\end{table}
""",
    {
        "VSAFE": f"{V_SAFE:.4f}",
        "VREF": f"{V_REF:.3f}",
        "ROWS": "\n".join(
            [
                res_row(r"public rows only", PUB),
                res_row(r"$+$ bucketed reveal pins ($K{=}4$)", GRP),
                res_row(r"$+$ per-combo reveal pins", PC),
            ]
        ),
    },
)
(TAB / "full_river_residual_table.tex").write_text(resid)


def fmt_gap(g):
    """Compute fmt gap for the render full river report workflow."""
    if g is None:
        return "---"
    m, e = f"{g:.0e}".split("e")
    return f"${m}\\times 10^{{{int(e)}}}$"


ee_lines = []
for r in ee["rows"]:
    if r["lp_value"] is None and r["lp_wall_s"] is None:
        continue
    inst = r["instance"].replace("holdem_tr_", "") + "/" + r["shape"]
    lp = (
        f"${r['lp_value']:.4f}$ (${r['lp_wall_s']:.0f}$\\,s)"
        if r["lp_value"] is not None
        else r"timeout ($600$\,s)"
    )
    ee_lines.append(
        f"{inst} & ${r['rows']}$ & {lp} & ${r['cp_value']:.4f}$ (${r['cp_wall_s']:.0f}$\\,s) & {fmt_gap(r['cp_gap'])} \\\\"
    )
solver = sub(
    r"""\begin{table}[t]
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\caption{Oracle decomposition vs.\ the monolithic robust LP on the bucketed
endgames ($200$-iteration cap, $600$\,s LP budget). \emph{Gap} is the
decomposition's own certificate (upper minus lower bound). Dense shapes reach
$10^{-4}$ parity; on the sparse b2 shape the monolithic LP times out while
the decomposition certifies; on sparse b4 the capped decomposition
under-claims honestly (its bracket contains the LP value).}
\label{tab:full_river_solver}
\begin{tabular}{@{}lrllr@{}}
\toprule
Instance & rows & monolithic LP & decomposition & gap \\
\midrule
@ROWS@
\bottomrule
\end{tabular}
\end{table}
""",
    {"ROWS": "\n".join(ee_lines)},
)
(TAB / "full_river_solver_table.tex").write_text(solver)


def dget(pair, n, arm):
    """Read a nested result field while preserving a default value."""
    return next(
        r
        for r in drift["rows"]
        if r["pair"] == pair and r["n_hands"] == n and r["arm"] == arm
    )


ARMN = {"static": "pooled", "windowed": "windowed", "cpub": r"$\Cpub$ pooled"}
d_lines = []
for i, n in enumerate(NS):
    for arm in ["static", "windowed", "cpub"]:
        r = dget("twin_off", n, arm)
        first = f"$10^{{{len(str(n)) - 1}}}$" if arm == "static" else ""
        d_lines.append(
            f"{first} & {ARMN[arm]} & ${r['cert_mean']:.3f}$ & ${r['real2_mean']:.3f}$ & "
            f"${r['overclaim_mean']:+.3f}$ & ${r['viol_y2_mean']:.3f}$ \\\\"
        )
    if i < len(NS) - 1:
        d_lines.append(r"\addlinespace[1.5pt]")
o7 = dget("overfold_shift", 10_000_000, "static")
w7 = dget("overfold_shift", 10_000_000, "windowed")
driftt = sub(
    r"""\begin{table}[t]
\centering\footnotesize
\setlength{\tabcolsep}{3.5pt}
\caption{Drift stress test: the public twin reverts to equilibrium at the
stream's midpoint ($10$ seeds; SE $\le0.004$ throughout).  \emph{Real.}\ is
realized value against the \emph{current} (post-switch) opponent;
\emph{overclaim} $=$ certified $-$ realized; \emph{viol} is the exact
coverage violation of the current opponent in the arm's set.  The pooled
certificate's overclaim \emph{grows} with budget while the floor holds in
every cell; the trailing-window arm re-certifies with overclaim ${\approx}0$;
pooled $\Cpub$ never detects the switch (the twin's defining property).
When the leak instead \emph{grows} mid-stream (over-fold $0.15\to0.5$,
$N{=}10^7$), pooling errs the other way---certified $@OF_C@$ vs realized
$@OF_R@$---and the window is again tight ($@WF_C@$ vs $@WF_R@$).}
\label{tab:full_river_drift}
\begin{tabular}{@{}llrrrr@{}}
\toprule
$N$ & evidence & cert. & Real. & overclaim & viol \\
\midrule
@ROWS@
\bottomrule
\end{tabular}
\end{table}
""",
    {
        "ROWS": "\n".join(d_lines),
        "OF_C": f"{o7['cert_mean']:.3f}",
        "OF_R": f"{o7['real2_mean']:.3f}",
        "WF_C": f"{w7['cert_mean']:.3f}",
        "WF_R": f"{w7['real2_mean']:.3f}",
    },
)
(TAB / "full_river_drift_table.tex").write_text(driftt)


def zrow(label, rs):
    """Normalize one population record for the generated report."""
    ex = [r["exploit_safe"] for r in rs]
    ps = [r["pub_share"] for r in rs if r.get("pub_share") is not None]
    gs = [r["grp_share"] for r in rs if r.get("grp_share") is not None]
    return (
        f"{label} & ${len(rs)}$ & ${st.median(ex):.2f}$ [{min(ex):.2f}, {max(ex):.2f}] & "
        f"${st.median(ps):.2f}$ & ${st.median(gs):.2f}$ \\\\"
    )


zr = [
    r
    for r in zoo["rows"]
    if r["exploit_safe"] > 0.02 and r.get("pub_share") is not None
]
z_lines = [
    zrow(
        r"sampled, strength $0.3$",
        [r for r in zr if r["class"] == "sampled" and r["param"] == 30],
    ),
    zrow(
        r"sampled, strength $0.5$",
        [r for r in zr if r["class"] == "sampled" and r["param"] == 50],
    ),
    zrow(
        r"sampled, strength $0.8$",
        [r for r in zr if r["class"] == "sampled" and r["param"] == 80],
    ),
    zrow(r"undertrained CFR$^+$", [r for r in zr if r["class"] == "weakcfr"]),
    r"\midrule",
    zrow(r"\textbf{all}", zr),
]
zoot = sub(
    r"""\begin{table}[t]
\centering\footnotesize
\setlength{\tabcolsep}{2.5pt}
\caption{A population of non-constructed opponents at the full unbucketed
river: hash-seeded spot-level fold/call perturbations at three strengths and
undertrained CFR$^+$ opponents.  Per class: opponents with safe-exploitable
value $V-\vref>0.02$, its median [min, max], and the median \emph{share} of
that value certified by the population public fiber ($R_{\mathrm{pub}}$) and
by $3$-probe bucketed reveal pins ($R_{\mathrm{grp}}$).  A typical sampled
opponent is about three-quarters public-certifiable and ${\sim}90\%$
reveal-certifiable (means $0.73$/$0.91$, matching the medians); the public
twin ($0\%$ public, $96\%$ reveal) is the constructed corner of this
distribution, not its typical member.  Three degenerate grouped fibers
enter as monotone brackets and are excluded from the reveal statistics.}
\label{tab:full_river_zoo}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Class & $n$ & $V{-}\vref$ med.\ [range] & public & reveal \\
\midrule
@ROWS@
\bottomrule
\end{tabular}
\end{table}
""",
    {"ROWS": "\n".join(z_lines)},
)
(TAB / "full_river_zoo_table.tex").write_text(zoot)

print(
    "wrote: fig_full_river_twin_ladder.tex, ed_full_river_deploy_table.tex, ed_full_river_perfamily_table.tex,"
)
print(
    "       full_river_residual_table.tex, full_river_solver_table.tex, full_river_drift_table.tex, full_river_zoo_table.tex"
)
print(
    f"audit: V_safe={V_SAFE:.4f} R_M(grp)={ASYM:.4f} v_ref={V_REF:.4f} "
    f"Delta-(pub)={PUB['delta_minus']:.4f} floor_violations={viol}"
)
