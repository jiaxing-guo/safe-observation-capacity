""

AVAILABLE_GAMES: tuple[str, ...] = ("kuhn",)


def is_available(name: str) -> bool:
    ""
    return name in AVAILABLE_GAMES


__all__ = ["AVAILABLE_GAMES", "is_available"]
