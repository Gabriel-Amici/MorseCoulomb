"""
Utilities for serialising and reconstructing emerald objects.

Types are stored by a stable string key and resolved lazily, so a file can be
loaded without importing every module that defines a serialisable class.
"""

from importlib import import_module

import numpy as np


class SerializationError(ValueError):
    """Raised when serialised data cannot be interpreted."""


_REGISTRY: dict[str, tuple[str, str]] = {
    "Coulomb": ("emerald.potentials.coulomb", "Coulomb"),
    "Morse": ("emerald.potentials.morse", "Morse"),
    "MorseSoftCoulomb": ("emerald.potentials.msc", "MorseSoftCoulomb"),
    "SoftCoulomb": ("emerald.potentials.soft_coulomb", "SoftCoulomb"),
}


def register(name: str, target: type | str) -> None:
    """
    Register a class under a stable serialisation key.

    Parameters
    ----------
    name : str
        Key stored in the ``_type`` field of serialised data.
    target : type or str
        Class object, or a ``"module:Class"`` string.
    """
    if isinstance(target, type):
        module_name = target.__module__
        class_name = target.__qualname__
    elif isinstance(target, str):
        module_name, separator, class_name = target.partition(":")
        if not separator:
            raise ValueError("String targets must have the form 'module:Class'.")
    else:
        raise TypeError("target must be a class or a 'module:Class' string.")

    existing = _REGISTRY.get(name)
    resolved = (module_name, class_name)
    if existing is not None and existing != resolved:
        raise ValueError(
            f"'{name}' is already registered as "
            f"{existing[0]}:{existing[1]}, cannot register {module_name}:{class_name}."
        )
    _REGISTRY[name] = resolved


def resolve(name: str) -> type:
    """Return the class registered under ``name``, importing its module lazily."""
    try:
        module_name, class_name = _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise SerializationError(
            f"Unknown serialisation type '{name}'. Registered types: {known}."
        ) from None

    obj = import_module(module_name)
    for part in class_name.split("."):
        obj = getattr(obj, part)
    return obj


def to_jsonable(value):
    """Convert NumPy scalars and arrays into plain JSON-compatible values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
