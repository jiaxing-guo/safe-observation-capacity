""

import json
import statistics as st
import sys

GAME = sys.argv[1] if len(sys.argv) > 1 else "holdem_tr_b2"
DEV = ["river_overfold_w80", "turn_overfold_w70", "revealed_call_strong"]
with open(f"results/sad_deploy_{GAME}.cells.jsonl") as _fh:
    cells = [json.loads(line) for line in _fh]


def C(mode, opp, N):
    return [
        c
        for c in cells
        if c["mode"] == mode
        and c["opponent"] == opp
        and c["episodes"] == N
        and not c.get("lp_timeout")
    ]


def cval(mode, opp, N, key="certified"):
    r = C(mode, opp, N)
    return st.mean(c[key] for c in r) if r else float("nan")


cp = {o: cval("cpub", o, 0) for o in DEV}
cpr = {o: cval("cpub", o, 0, "realized") for o in DEV}
orc = {o: cval("oracle", o, 0) for o in DEV}
orr = {o: cval("oracle", o, 0, "realized") for o in DEV}


def cap(cert_by_fam):

    return st.mean(
        (cert_by_fam[o] - cp[o]) / (orr[o] - cp[o]) for o in DEV if orr[o] > cp[o]
    )


print(f"=== {GAME}: tab:deploy rows (N=1e5, mean over 3 dev families) ===")
print(f"{'mode':16s} {'cert':>6s} {'real':>6s} {'cap':>6s} {'infeas':>7s} {'viol':>6s}")
rows = {}
for mode in ["cpub", "passive", "random", "sad", "oracle_target", "oracle"]:
    N = 100000 if mode not in ("cpub", "oracle") else 0
    certf = {o: cval(mode, o, N) for o in DEV}
    realf = {o: cval(mode, o, N, "realized") for o in DEV}
    inf = sum(c.get("box_infeasible", False) for o in DEV for c in C(mode, o, N))
    tot = sum(len(C(mode, o, N)) for o in DEV)
    vio = sum(c.get("violation", False) for o in DEV for c in C(mode, o, N))
    cert = st.mean(certf.values())
    real = st.mean(realf.values())
    capv = cap(certf)
    rows[mode] = (cert, real, capv, certf, realf)
    infs = f"{inf}/{tot}" if N else "--"
    print(f"{mode:16s} {cert:6.3f} {real:6.3f} {capv:6.2f} {infs:>7s} {vio:>4d}/{tot}")


print()
print("=== Routed SAD (per-cell argmax of cpub vs sad), by N ===")
for N in [1000, 10000, 100000]:
    rc, rr, rcf = [], [], {}
    for o in DEV:
        cc, rcell = [], []
        for c in C("sad", o, N):
            if c["certified"] >= cp[o]:
                cc.append(c["certified"])
                rcell.append(c["realized"])
            else:
                cc.append(cp[o])
                rcell.append(cpr[o])
        rcf[o] = st.mean(cc)
        rc.append(st.mean(cc))
        rr.append(st.mean(rcell))
    capv = cap(rcf)
    vio = sum(c.get("violation", False) for o in DEV for c in C("sad", o, N))
    tot = sum(len(C("sad", o, N)) for o in DEV)
    print(
        f"  N={N:6d}: cert {st.mean(rc):.3f}  real {st.mean(rr):.3f}  "
        f"cap {capv:.2f}  viol {vio}/{tot}"
    )

print()
print("=== per-family certified (cpub | sad@1e3/1e4/1e5 | oTgt@1e5 | oracle) ===")
for o in DEV:
    print(
        f"  {o:22s} cpub {cp[o]:.3f} | sad {cval('sad', o, 1000):.3f}/"
        f"{cval('sad', o, 10000):.3f}/{cval('sad', o, 100000):.3f} | "
        f"oTgt {cval('oracle_target', o, 100000):.3f} | oracle {orc[o]:.3f}"
    )

print()
print("=== infeasibility passive vs sad @1e5 (per family) ===")
for o in DEV:
    pi = sum(c.get("box_infeasible", False) for c in C("passive", o, 100000))
    pn = len(C("passive", o, 100000))
    si = sum(c.get("box_infeasible", False) for c in C("sad", o, 100000))
    sn = len(C("sad", o, 100000))
    print(f"  {o:22s} passive {pi}/{pn}  sad {si}/{sn}")

tot_cells = len(cells)
tmo = sum(c.get("lp_timeout", False) for c in cells)
allvio = sum(c.get("violation", False) for c in cells)
print(f"\ntotal cells {tot_cells} | lp_timeout {tmo} | violations {allvio}")
