"""Blueprint primitives for safe-observation experiments. See Safe Active De-censoring, Experiments, and supplementary Game Instances and Experimental Setup."""

from pathlib import Path
from typing import Any

from .. import evaluation, solvers

KNOWN_KUHN_VALUE = -1.0 / 18.0


def run_kuhn_blueprint(
    method: str = "lp", iterations: int = 100_000, out_dir: str | Path = "results"
) -> dict[str, Any]:
    """Run the Kuhn blueprint experiment."""
    solution = solvers.solve_blueprint("kuhn", method=method, iterations=iterations)
    results: dict[str, Any] = {
        "game": "kuhn",
        "method": solution.method,
        "iterations": iterations if method == "cfr" else None,
        "value_player1": solution.value,
        "known_value": KNOWN_KUHN_VALUE,
        "abs_error": abs(solution.value - KNOWN_KUHN_VALUE),
        "strategy": solution.strategy,
    }
    if solution.realization is not None:
        safety = solvers.safety_verifier(solution.realization, game="kuhn")
        results["safety_floor"] = safety.value
        results["safety_floor_gap"] = abs(safety.value - solution.value)
    evaluation.save_results(results, Path(out_dir) / "kuhn_blueprint.json")
    return results


__all__ = ["run_kuhn_blueprint", "KNOWN_KUHN_VALUE"]
