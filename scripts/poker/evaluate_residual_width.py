""

from __future__ import annotations

import sys

import numpy as np
from scipy.optimize import linprog

from safe_observation.opponents import (
    leduc_static_biased_opponent,
    static_biased_opponent,
)
from safe_observation.payoff import PayoffMatrix
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import (
    robust_safe_response_linear,
    safety_constrained_best_response,
    solve_blueprint,
)

GAME = sys.argv[1] if len(sys.argv) > 1 else "leduc"
RHO = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
TOL = 1e-6


def _opponent(game: str):
    ""
    if game == "leduc":
        return leduc_static_biased_opponent()
    if game == "kuhn":
        return static_biased_opponent(bet_prob=0.1)
    raise SystemExit(f"unsupported game {game!r} (use leduc or kuhn)")


def _continuations(sf, y_star):
    ""
    out = []
    for info in sf.info_sets:
        acts = [a for a, _ in info.children]
        fi = acts.index("f") if "f" in acts else None
        parent = y_star[info.parent_seq]
        for a, (_act, child) in enumerate(info.children):
            if a == fi:
                continue
            out.append((info.label, a, child, y_star[child], parent))
    return out


def _pin_rows(pins):
    ""
    entries: list[tuple[int, int, float]] = []
    h: list[float] = []
    meta: list[tuple[str, int]] = []
    row = 0
    for child, val in pins:
        entries.append((row, child, 1.0))
        h.append(val)
        meta.append((f"pin{child}", 0))
        row += 1
        entries.append((row, child, -1.0))
        h.append(-val)
        meta.append((f"pin{child}", 0))
        row += 1
    return entries, h, meta


def _R_M(pins, v_ref):
    ""
    entries, h, meta = _pin_rows(pins)
    res = robust_safe_response_linear(
        {}, {}, entries, h, v_ref=v_ref, eps_safe=RHO, game=GAME, row_meta=meta
    )
    return res.robust_value


def _opp_constraints(sf):
    ""
    m = sf.num_constraints if hasattr(sf, "num_constraints") else len(sf.e)
    n = sf.num_sequences
    E = np.zeros((m, n))
    for r, c, v in sf.e_entries:
        E[r, c] += v
    e = np.asarray(sf.e, dtype=float)
    return E, e


def _fiber_argmin(sf, c, forced_children, y_star):
    ""
    E, e = _opp_constraints(sf)
    pin_rows = []
    pin_rhs = []
    for child in forced_children:
        row = np.zeros(sf.num_sequences)
        row[child] = 1.0
        pin_rows.append(row)
        pin_rhs.append(y_star[child])
    A_eq = np.vstack([E, *pin_rows]) if pin_rows else E
    b_eq = np.concatenate([e, np.asarray(pin_rhs)]) if pin_rows else e
    res = linprog(
        c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0.0, None)] * sf.num_sequences,
        method="highs",
    )
    if not res.success:
        return None
    return np.asarray(res.x)


def main() -> None:
    print(f"# Residual-width probe on {GAME}  (rho={RHO})\n")
    sf = compile_game(GAME, 1)
    bp = solve_blueprint(GAME, method="lp")
    v_ref = bp.value
    P = PayoffMatrix(GAME)
    opp = _opponent(GAME)
    y_star = list(opp.realization())
    behavior = opp.behavior

    print(
        f"v_ref={v_ref:.6f}  n_seq_p2={sf.num_sequences}  n_info_p2={sf.num_infosets}"
    )

    scbr = safety_constrained_best_response(
        behavior, v_ref=v_ref, eps_safe=RHO, game=GAME
    )
    V = scbr.value
    x_V = list(scbr.realization)
    print(
        f"V(y*) = max_x in S(rho) x^T A y* = {V:.6f}   (exploitation {V - v_ref:+.6f})"
    )

    conts = _continuations(sf, y_star)
    all_children = [c[2] for c in conts]
    print(f"non-fold continuations: {len(conts)}\n")

    print("== Partial-forcing feasibility ==")
    R_full = _R_M([(c[2], c[3]) for c in conts], v_ref)
    print(
        f"FULL forcing  R_full = {R_full:.6f}   shortfall V-R_full = {V - R_full:+.2e}"
    )
    if abs(V - R_full) < 1e-4:
        print("  -> point-identification recovered: pin/index construction OK")
    else:
        print("  -> WARNING: full forcing != V; check indexing before trusting drops")

    drops = []
    for label, a, child, _val, parent in conts:
        if parent <= TOL:
            continue
        kept = [(c[2], c[3]) for c in conts if c[2] != child]
        R_drop = _R_M(kept, v_ref)
        drops.append((V - R_drop, label, a, child, parent))
    drops.sort(reverse=True)

    print("\nPartial forcing -- drop ONE continuation, residual shortfall V - R_M:")
    print(f"  {'shortfall':>11}  {'parent_m':>9}  infoset/action")
    n_pos = sum(1 for d in drops if d[0] > 1e-5)
    for sh, label, a, _child, parent in drops[:12]:
        flag = " *" if sh > 1e-5 else "  "
        print(f"  {sh:>11.6f}  {parent:>9.4f}  {label} [a{a}]{flag}")
    print(
        f"\n  {n_pos}/{len(drops)} reached continuations have POSITIVE residual "
        f"shortfall when unforced"
    )
    print(
        "  (positive = value-relevant -> would need forcing; "
        "zero = payoff-null residual, safe to leave unforced)"
    )

    if not drops or drops[0][0] <= 1e-5:
        print(
            "\n[residual ambiguity analysis skipped] no value-relevant residual at this rho; "
            "try a larger rho or a different opponent."
        )
        _verdict(feasible=abs(V - R_full) < 1e-4, residual=False)
        return

    print(
        "\n== residual ambiguity: subspace-sup vacuity vs corrected exact shortfall =="
    )
    top = drops[0]
    _, top_label, top_a, top_child, _ = top
    forced = [ch for ch in all_children if ch != top_child]
    print(f"partial protocol: force all EXCEPT  {top_label} [a{top_a}]")
    print(f"  exact fiber shortfall  V - R_M = {top[0]:.6f}   (finite, LP-computable)")

    c_obj = np.asarray(P.matvec_at_x(x_V))
    y_min = _fiber_argmin(sf, c_obj, forced, y_star)
    if y_min is None:
        print("  fiber-min LP failed; cannot extract residual direction")
        _verdict(feasible=abs(V - R_full) < 1e-4, residual=True)
        return
    d = np.asarray(y_star) - y_min

    E, _e = _opp_constraints(sf)
    flow_resid = float(np.max(np.abs(E @ d)))
    forced_resid = float(max((abs(d[ch]) for ch in forced), default=0.0))

    x_bp = list(bp.realization)
    s_xV = float(P.bilinear(x_V, list(d)))
    s_bp = float(P.bilinear(x_bp, list(d)))
    spread_lb = max(s_xV, s_bp, 0.0) - min(s_xV, s_bp, 0.0)
    print(
        f"  residual d=y*-y_min in N_res:  ||E_2 d||inf={flow_resid:.2e}  "
        f"max|d on forced|={forced_resid:.2e}"
    )
    print(
        f"  payoff spread lower bound eta(d) >= {spread_lb:.6f}  "
        f"(x_V^T A d={s_xV:+.6f}, x_bp^T A d={s_bp:+.6f})"
    )

    if spread_lb <= 1e-6:
        print("  spread ~ 0 here -> this residual is (numerically) payoff-null")
    else:
        print(
            "\n  OLD bound  sup_{d in N_res} eta(d)  scales linearly in ||d|| (VACUOUS):"
        )
        for lam in (1, 2, 10, 100):
            print(f"     eta({lam:>3} d) >= {lam * spread_lb:.6f}")
        print(f"  NEW bound  exact shortfall V - R_M = {top[0]:.6f}  (finite, correct)")
        print(f"     one-sided witness Delta^-_M(y*) >= x_V^T A d = {s_xV:.6f}")

    _verdict(feasible=abs(V - R_full) < 1e-4, residual=spread_lb > 1e-6)


def _verdict(feasible: bool, residual: bool) -> None:
    print("\n== verdict ==")
    print(
        f"  partial-forcing fiber R_M computable and full-forcing equals V: "
        f"{'PASS' if feasible else 'FAIL'}"
    )
    print(
        f"  non-point residual with positive payoff spread exhibited : "
        f"{'YES' if residual else 'no (payoff-null at this rho/opp)'}"
    )
    print("  bound fix: exact V-R_M is the finite object; subspace sup is vacuous.")


if __name__ == "__main__":
    main()
