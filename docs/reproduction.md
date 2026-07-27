# Reproduction guide

## Small reference runs

```bash
uv run safe-observation run configs/kuhn/blueprint.toml
uv run safe-observation run configs/kuhn/static_opponent.toml
uv run safe-observation run configs/leduc/confidence_coverage.toml
```

## Showdown censoring

```bash
uv run safe-observation run configs/leduc/showdown_censoring.toml
uv run safe-observation run configs/holdem/showdown_censoring.toml
uv run safe-observation run configs/holdem/showdown_sweep.toml
```

## Safe active de-censoring

The principal controller entry point is `scripts/poker/run_safe_active_decensoring.py`.
Each empirical cell charges one total budget: public-only collection spends it under
the blueprint, while active arms use a 20% blueprint pilot and a disjoint 80% reveal
batch. The route is frozen before the reveal batch.

Run the headline grid with:

```bash
SAD_MODES=cpub,random,sad,oracle_target,oracle \
uv run python -m scripts.poker.run_safe_active_decensoring \
  holdem_tr_b2 0.5 1000000 10 10 600
```

Run the matched-budget crossover study with:

```bash
SAD_MODES=cpub,random,sad \
uv run python -m scripts.poker.run_safe_active_decensoring \
  holdem_tr_b2 0.5 100000,300000,1000000 10 10 600
```

Related mechanism and scope entry points are
`scripts/poker/run_active_decensoring.py`,
`scripts/poker/run_identification_coverage.py`,
`scripts/poker/run_observation_capacity_frontier.py`,
`scripts/poker/run_opponent_population.py`,
`scripts/poker/audit_residual_ambiguity.py`, and
`scripts/poker/run_turn_river_methods.py`.

## Controlled instances

The `scripts/controlled` directory contains bandit, MDP, censored-chain, and selective-label studies used to test which conclusions depend on poker structure.

## Reporting

Reporting scripts consume locally generated JSON or JSONL data and write tables or figures beneath the ignored `generated` directory.

## Verification

```bash
make test-fast
make check
```

Ignored long-running Rust tests can be selected explicitly with Cargo when needed. No scheduler-specific wrapper is required.
