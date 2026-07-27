"""Game construction interfaces used by configured experiments."""

AVAILABLE_GAMES: tuple[str, ...] = ("kuhn",)


def is_available(name: str) -> bool:
    """Return whether available for the init workflow."""
    return name in AVAILABLE_GAMES


__all__ = ["AVAILABLE_GAMES", "is_available"]
