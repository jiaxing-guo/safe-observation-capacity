"""Validate lower bound. See supplementary Additional Experiments."""

from collections import defaultdict

from safe_observation.opponents import Opponent
from safe_observation.payoff import build as build_payoff
from safe_observation.sequence_form import compile as compile_game
from safe_observation.solvers import (
    agent_showdown_reach,
    opponent_reach_weights,
    robust_safe_response_probe,
    safety_constrained_best_response,
    solve_blueprint,
)

GAME = "kuhn"
TARGET = ["0:b", "2:b"]
EPS = [0.2, 0.1, 0.05, 0.025]
RHOS = [0.1, 0.3]


def _base(sf1):
    """Compute base for the validate lower bound workflow."""
    return {i.label: [0.5, 0.5] for i in sf1.info_sets}


def _two_point(sf1, eps):
    """Compute two point for the validate lower bound workflow."""
    y0, y1 = _base(sf1), _base(sf1)

    y0[TARGET[0]] = [0.5 + eps, 0.5 - eps]
    y0[TARGET[1]] = [0.5 - eps, 0.5 + eps]
    y1[TARGET[0]] = [0.5 - eps, 0.5 + eps]
    y1[TARGET[1]] = [0.5 + eps, 0.5 - eps]
    return y0, y1


def _real(beh):
    """Compute real for the validate lower bound workflow."""
    return list(Opponent(name="x", behavior=beh, game=GAME).realization())


def _tv_public(sf1, y0r, y1r):
    """Compute tv public for the validate lower bound workflow."""

    def summed(yr):
        """Compute summed for the validate lower bound workflow."""
        out = defaultdict(float)
        for info in sf1.info_sets:
            hist = info.label.split(":", 1)[1]
            for a, (_ac, child) in enumerate(info.children):
                out[(hist, a)] += yr[child]
        return out

    s0, s1 = summed(y0r), summed(y1r)
    return 0.5 * sum(abs(s0[k] - s1[k]) for k in set(s0) | set(s1))


def _reveal_diff(x_real, y0r, y1r, sf1):
    """Compute reveal diff for the validate lower bound workflow."""
    sd = agent_showdown_reach(x_real, game=GAME)
    by_label = {i.label: i for i in sf1.info_sets}
    tot = 0.0
    kap = 0.0
    for lbl in TARGET:
        row = sd.get(lbl)
        info = by_label[lbl]
        ci = [a for a, _ in info.children].index("b")
        if row is None or ci >= len(row):
            continue
        w_sd, _committing = row[ci]
        child = info.children[ci][1]
        tot += abs(w_sd * (y0r[child] - y1r[child]))
        kap += w_sd
    return tot, kap


def main() -> None:
    """Run the command-line entry point."""
    sf1 = compile_game(GAME, 1)
    payoff = build_payoff(GAME)
    bp = solve_blueprint(GAME, method="lp")
    v_ref = bp.value
    x_bp = list(bp.realization)
    triv_iv = {i.label: [(0.0, 1.0)] * len(i.children) for i in sf1.info_sets}

    cont_beh = {}
    for info in sf1.info_sets:
        if info.label.endswith(":b"):
            cont_beh[info.label] = [0.0, 1.0]
        else:
            cont_beh[info.label] = [0.5, 0.5]

    print(f"# Lower-bound two-point spike on {GAME}  (v_ref={v_ref:.4f})")
    print(f"# target={TARGET}\n")
    print(
        f"{'rho':>5}{'eps':>7}{'gap':>9}{'gap/eps':>8}{'tvpub':>8}"
        f"{'kappa':>8}{'d_act':>9}{'d_pas':>9}{'Ncert':>10}{'N*lam*e2':>10}"
    )
    for rho in RHOS:
        for eps in EPS:
            y0b, y1b = _two_point(sf1, eps)
            y0r, y1r = _real(y0b), _real(y1b)
            tvpub = _tv_public(sf1, y0r, y1r)

            r0 = safety_constrained_best_response(
                y0b, v_ref=v_ref, eps_safe=rho, game=GAME
            )
            r1 = safety_constrained_best_response(
                y1b, v_ref=v_ref, eps_safe=rho, game=GAME
            )
            v00 = payoff.bilinear(list(r0.realization), y0r)
            v10 = payoff.bilinear(list(r1.realization), y0r)
            gap = abs(v00 - v10)

            probe = robust_safe_response_probe(
                triv_iv,
                cont_beh,
                {t: 1.0 for t in TARGET},
                v_ref=v_ref,
                eps_safe=rho,
                beta=1e6,
                rho=0.0,
                game=GAME,
            )
            xpr = list(probe.realization)
            d_act, kap = _reveal_diff(xpr, y0r, y1r, sf1)
            d_pas, _ = _reveal_diff(x_bp, y0r, y1r, sf1)
            lam = max(kap, 1e-12)
            ncert = lam / max(d_act, 1e-12) ** 2
            check = ncert * lam * eps**2
            print(
                f"{rho:>5.2f}{eps:>7.3f}{gap:>9.4f}{gap / eps:>8.3f}{tvpub:>8.1e}"
                f"{kap:>8.3f}{d_act:>9.4f}{d_pas:>9.1e}{ncert:>10.1f}{check:>10.3f}"
            )
        print()
    _ = opponent_reach_weights


if __name__ == "__main__":
    main()
