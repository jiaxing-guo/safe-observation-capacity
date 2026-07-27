"""Public interfaces for evaluation. See supplementary Reproducibility for its role in the release workflow."""

import json
from pathlib import Path
from typing import Any


def save_results(results: dict[str, Any], path: str | Path) -> Path:
    """Save results for the init workflow."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    return out


__all__ = ["save_results"]
