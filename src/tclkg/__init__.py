"""Temporal Constraint Learning for Knowledge Graphs."""

from importlib import import_module


_EXPORTED_MODULES = {
    "allen_list",
    "allen_relations",
    "qcn_generator2",
    "random_experiments",
    "rule_discovery",
    "launcher",
    "time_package",
}

__all__ = [
    "allen_list",
    "allen_relations",
    "qcn_generator2",
    "random_experiments",
    "rule_discovery",
    "launcher",
    "time_package",
]


def __getattr__(name: str):
    """Lazily expose submodules on first access.

    This avoids importing executable modules during package import,
    which prevents runpy warnings for `python -m tclkg.<module>`.
    """
    if name in _EXPORTED_MODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
