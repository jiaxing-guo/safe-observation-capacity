# Architecture

The project uses Python for orchestration and Rust for the computational core.

## Rust core

`crates/safe-observation-core` contains game trees, sequence-form compilation, CFR solvers, payoff construction, confidence-set constraints, robust response optimization, observation-capacity computations, and simulation. The core can be built and tested without Python.

`crates/safe-observation-python` exposes the Rust API as `safe_observation_native` through PyO3.

## Python package

`src/safe_observation` provides typed wrappers around the native extension, opponent families, evidence stores, floor-safe agents, experiment configuration, and reporting helpers.

The public interface is grouped by responsibility:

- `games`, `sequence_form`, and `payoff` define game representations.
- `confidence` constructs private and public observation sets.
- `solvers` computes blueprints, best responses, floor-safe responses, and observation-capacity plans.
- `agents` and `probe` implement online collection and safety-budget accounting.
- `experiments` runs configured evaluations.

## Experiment boundary

Reusable library code lives under `src` and `crates`. One-off study entry points live under `scripts`, separated into poker studies, controlled instances, reporting, and validation. Configurations are grouped by game rather than by internal development milestone.

Generated data is written to ignored directories.
