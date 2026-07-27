"""Public interfaces for safe observation. See supplementary Reproducibility for its role in the release workflow."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from . import native

try:
    __version__ = _pkg_version("safe-observation-capacity")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["native", "__version__"]
