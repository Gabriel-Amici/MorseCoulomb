"""
Module for basic field definition
"""

import numpy as np

from ..numerics.maths import AnalyticFunction, GridFunction
from ..serialization import Serializable, SerializationError


class Field(AnalyticFunction, Serializable):
    """
    Basic (electric) field class, given an array of field values and times, calling it results in an
    evaluation of the field (interpolated when necessary) at the specified time
    """

    _serialized_parameters: dict[str, str] = {}
    _format_version = 1

    def __init__(self):
        pass

    @classmethod
    def from_dict(cls, data: dict) -> "Field":
        version = data.get("_format_version")
        if version != cls._format_version:
            raise SerializationError(
                f"Unsupported field format version: {version}."
            )

        type_name = data.get("_type")
        if type_name is None:
            raise SerializationError("Field data is missing '_type'.")

        from ..serialization import resolve

        concrete = resolve(type_name)
        if not (issubclass(concrete, Field) or issubclass(concrete, InterpolatedField)):
            raise SerializationError(
                f"Serialisation type '{type_name}' is not a Field subclass."
            )

        return Serializable.from_dict.__func__(concrete, data)


class InterpolatedField(GridFunction, Serializable):

    _serialized_parameters = {"field_times": "field_times", "field_vals": "field_vals"}
    _format_version = 1

    def __init__(self, field_times: np.ndarray, field_vals: np.ndarray):
        super().__init__(field_times, field_vals)
        if len(field_times) != len(field_vals):
            raise Warning(f"Time array and field array must be of same size! "
                          f"{len(field_times)} ≠ {len(field_vals)}")

        self.field_times = np.asarray(field_times)
        self.field_vals = np.asarray(field_vals)