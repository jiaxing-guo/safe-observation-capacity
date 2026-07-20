# Reproduction guide

The release contains source and experiment definitions but no generated results, figures, logs, review artifacts, or cluster submission files.

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

The principal study entry points are:

- `scripts/poker/run_safe_active_decensoring.py`
- `scripts/poker/run_active_decensoring.py`
- `scripts/poker/run_identification_coverage.py`
- `scripts/poker/run_observation_capacity_frontier.py`
- `scripts/poker/run_opponent_population.py`
- `scripts/poker/audit_residual_ambiguity.py`
- `scripts/poker/run_turn_river_methods.py`

Run a script from the repository root as a module, for example:

```bash
uv run python -m scripts.poker.run_safe_active_decensoring
```

Scripts expose their experiment sizes through command-line arguments or environment variables and write outputs beneath ignored local directories.

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
