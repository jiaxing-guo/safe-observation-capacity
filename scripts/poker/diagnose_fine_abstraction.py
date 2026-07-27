"""Diagnose fine abstraction. See Experiments and supplementary Certification at the Unbucketed River."""

import sys

sys.argv_backup = list(sys.argv)
GAME = sys.argv[1] if len(sys.argv) > 1 else "holdem_tr_b4"
OPP = sys.argv[2] if len(sys.argv) > 2 else "turn_overfold_w70"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 2026


sys.argv = ["run_safe_active_decensoring.py", GAME, "0.5", str(N), "1", "1", "600"]
from safe_observation.opponents import Opponent
from scripts.poker import run_safe_active_decensoring as D


def concentration(omega):
    """Compute concentration for the diagnose fine abstraction workflow."""
    vals = sorted((v for v in omega.values() if v > 0), reverse=True)
    if not vals:
        return float("nan")
    tot = sum(vals)
    k = max(1, len(vals) // 10)
    return sum(vals[:k]) / tot


def main():
    """Run the command-line entry point."""
    D._init()
    sf1, groups, info_by = D._W["sf1"], D._W["groups"], D._W["info_by"]
    v_ref, omega_bp, fold_idx = D._W["v_ref"], D._W["omega_bp"], D._W["fold_idx"]
    behavior = D._W["suite"][OPP]
    y_star = list(Opponent(name=OPP, behavior=behavior, game=GAME).realization())

    pop = D._population_public_intervals(groups, info_by, y_star, omega_bp)
    cpub = D.robust_safe_response_public(
        groups, pop, v_ref=v_ref, eps_safe=D.RHO, game=GAME, weights=omega_bp
    )
    n_groups = len(groups)
    print(f"# {GAME}  {OPP}  N={N}  seed={SEED}  (rho={D.RHO})")
    print(f"  v_ref={v_ref:.4f}  floor={v_ref - D.RHO:.4f}  #public_groups={n_groups}")
    print(f"  CPUB (population C_pub):  certified={cpub.robust_value - v_ref:+.4f}\n")
    print(
        f"  {'mode':14s}{'probe_safe':>11}{'dev':>7}{'reach_top10%':>13}"
        f"{'empty_grp':>10}{'pins':>6}{'pubOnly':>9}{'pub+evt':>9}"
    )

    for mode in ("random", "sad", "oracle_target"):
        key, beh, sv = D._probe_task((mode, OPP))
        agent_behavior = beh if beh is not None else D._W["bp_behavior"]
        agent_real = D._W["sf0"].realization_from_behavior(agent_behavior)
        omega = D.opponent_reach_weights(agent_real, game=GAME)

        _pay, show, fold = D.native.simulate_showdown(
            GAME, agent_behavior, behavior, N, SEED
        )
        ev = D._Store.for_game(GAME)
        for lab, c in show.items():
            ev.record(lab, c)
        for lab, c in fold.items():
            ev.record(lab, c)
        pub_intervals = ev.public_intervals(D.DELTA, method=D.METHOD)

        empty = sum(
            1
            for k, labs in groups.items()
            if not any(ev.visits(lbl) > 0 for lbl in labs)
        )

        sd_reach = D.agent_showdown_reach(agent_real, game=GAME)
        ev_entries, ev_h, ev_meta, n_pairs = D._empirical_mass_constraints(
            sf1, show, sd_reach, fold_idx, N, D.DELTA, D.METHOD
        )

        pub_only = D._try_solve(
            lambda pi=pub_intervals, w=omega: D.robust_safe_response_public(
                groups,
                pi,
                v_ref=v_ref,
                eps_safe=D.RHO,
                game=GAME,
                weights=w,
            )
        )

        full = D._try_solve(
            lambda pi=pub_intervals, e=ev_entries, h=ev_h, w=omega, m=ev_meta: (
                D.robust_safe_response_linear(
                    groups,
                    pi,
                    e,
                    h,
                    v_ref=v_ref,
                    eps_safe=D.RHO,
                    game=GAME,
                    weights=w,
                    row_meta=m,
                )
            )
        )
        dev = (v_ref - D.RHO) - sv
        po = (pub_only.robust_value - v_ref) if pub_only else float("nan")
        fu = (full.robust_value - v_ref) if full else float("nan")
        print(
            f"  {mode:14s}{sv:>11.4f}{dev:>7.3f}{concentration(omega):>13.3f}"
            f"{empty:>7}/{n_groups:<3}{n_pairs:>6}{po:>+9.3f}{fu:>+9.3f}"
        )

    print(
        "\n  KEY: pubOnly < CPUB  => empirical box looser than population "
        "(coverage holes from targeting)."
    )
    print("       pub+evt - pubOnly  => what the committing event box recovers.")


if __name__ == "__main__":
    main()
