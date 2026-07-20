# Safe Observation Capacity

Research code for safe active de-censoring in two-player zero-sum imperfect-information games.

The project studies how an agent can actively reveal behavior hidden by showdown censoring while preserving a worst-case value floor. It includes sequence-form game models, confidence sets, floor-safe response solvers, observation-capacity experiments, and evaluations on Kuhn poker, Leduc Hold'em, Goofspiel, and heads-up no-limit Hold'em endgames.

## Repository layout

```text
crates/
  safe-observation-core/      Rust game models, CFR, sequence form, LPs, and simulators
  safe-observation-python/    PyO3 extension module
src/safe_observation/         Python API and experiment orchestration
configs/
  kuhn/                       Kuhn reference experiments
  leduc/                      Leduc safety and censoring experiments
  holdem/                     Hold'em showdown-censoring experiments
  goofspiel/                  Non-poker generality experiments
scripts/
  poker/                      Poker experiments and population studies
  controlled/                 Controlled bandit, MDP, and selective-label studies
  reporting/                  Table and figure generation
  validation/                 Independent mechanism and invariant checks
tests/                        Python integration and regression tests
```

Generated results, figures, logs, review artifacts, and cluster job definitions are intentionally absent from this repository.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Rust 1.83 or newer
- CMake and a C++ compiler for the HiGHS backend

## Installation

```bash
make install
```

## Quick start

```bash
uv run safe-observation info
make smoke
```

```python
from safe_observation import solvers

solution = solvers.solve_blueprint("kuhn")
print(solution.value)
```

## Reproduction

Configurations are grouped by game. For example:

```bash
uv run safe-observation run configs/kuhn/blueprint.toml
uv run safe-observation run configs/leduc/confidence_coverage.toml
uv run safe-observation run configs/holdem/showdown_censoring.toml
```

The experiment scripts write generated data to ignored local directories. See [docs/reproduction.md](docs/reproduction.md) for the release-facing experiment map.

## Verification

```bash
make test-fast
make check
```

## Responsible use

This repository is intended for offline research, solver development, and bot-versus-bot evaluation. It must not be used to automate real-money play, connect to online poker clients, or evade platform safeguards. See [docs/responsible-use.md](docs/responsible-use.md).

## License

Licensed under either the Apache License 2.0 or the MIT License, at your option.
