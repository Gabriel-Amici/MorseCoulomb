"""
Module for basic field definition
"""

import numpy as np

from ..numerics.maths import AnalyticFunction, GridFunction


class Field(AnalyticFunction):
    """
    Basic (electric) field class, given an array of field values and times, calling it results in an
    evaluation of the field (interpolated when necessary) at the specified time
    """

    def __init__(self):
        pass
    

class InterpolatedField(GridFunction):

    def __init__(self, field_times: np.ndarray, field_vals: np.ndarray):
        super().__init__(field_times, field_vals)
        if len(field_times) != len(field_vals):
            raise Warning(f"Time array and field array must be of same size! "
                          f"{len(field_times)} ≠ {len(field_vals)}")

        self.field_times = field_times
        self.field_vals  = field_vals