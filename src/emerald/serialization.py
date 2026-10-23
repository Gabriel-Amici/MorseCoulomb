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
    "InterpolatedField": ("emerald.fields.base", "InterpolatedField"),
    "SinusoidField": ("emerald.fields.sinusoidal", "SinusoidField"),
    "Dynamics": ("emerald.classical.dynamics", "Dynamics"),
    "StaticDynamics": ("emerald.classical.dynamics", "StaticDynamics"),
    "RegularizedDynamics": ("emerald.classical.dynamics", "RegularizedDynamics"),
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
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def array_to_dict(array) -> dict:
    array = np.asarray(array)
    return {
        "_type": "ndarray",
        "dtype": str(array.dtype),
        "shape": array.shape,
        "data": array.ravel().tolist(),
    }


def array_from_dict(data) -> np.ndarray:
    arr = np.array(data["data"], dtype=data["dtype"])
    return arr.reshape(data["shape"])


def serialize_object(obj):
    if obj is None:
        return None
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return to_jsonable(obj)


def deserialize_object(data):
    if data is None:
        return None
    if isinstance(data, dict):
        if data.get("_type") == "ndarray":
            return array_from_dict(data)
        if "_type" in data and "_format_version" in data:
            concrete = resolve(data["_type"])
            if hasattr(concrete, "from_dict"):
                return concrete.from_dict(data)
            raise SerializationError(
                f"'{data['_type']}' has no from_dict method."
            )
    return data


class Serializable:
    """Mixin for objects whose constructor arguments fully define their state."""

    _serialized_parameters: dict[str, str] = {}
    _format_version = 1

    def to_dict(self) -> dict:
        data = {
            "_type": type(self).__name__,
            "_format_version": self._format_version,
        }
        for parameter, attribute in self._serialized_parameters.items():
            value = getattr(self, attribute)
            if hasattr(value, "to_dict"):
                data[parameter] = value.to_dict()
            elif isinstance(value, np.ndarray):
                data[parameter] = array_to_dict(value)
            else:
                data[parameter] = to_jsonable(value)
        return data

    @classmethod
    def from_dict(cls, data: dict):
        version = data.get("_format_version")
        if version != cls._format_version:
            raise SerializationError(
                f"Unsupported format version: {version}."
            )

        type_name = data.get("_type")
        if type_name is None:
            raise SerializationError("Serialised data is missing '_type'.")

        concrete = resolve(type_name)
        if not issubclass(concrete, Serializable):
            raise SerializationError(
                f"Serialisation type '{type_name}' is not serialisable."
            )

        parameters = {}
        for parameter in concrete._serialized_parameters:
            if parameter not in data:
                raise SerializationError(
                    f"Missing parameter '{parameter}' for {type_name}."
                )
            parameters[parameter] = deserialize_object(data[parameter])

        return concrete(**parameters)
