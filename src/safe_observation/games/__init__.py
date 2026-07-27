"""Public interfaces for games. See supplementary Reproducibility for its role in the release workflow."""

AVAILABLE_GAMES: tuple[str, ...] = ("kuhn",)


def is_available(name: str) -> bool:
    """Return whether available for the init workflow."""
    return name in AVAILABLE_GAMES


__all__ = ["AVAILABLE_GAMES", "is_available"]
