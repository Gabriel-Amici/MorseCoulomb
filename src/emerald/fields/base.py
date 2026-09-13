"""
Module for basic field definition
"""

import numpy as np


class Field:
    """
    Basic (electric) field class, given an array of field values and times, calling it results in an
    evaluation of the field (interpolated when necessary) at the specified time
    """

    def __init__(self):
        pass
    
    def __call__(self, times):
        return self.value(times)

    def value(self, times):
        """
        Evaluate the field at `times`
        """

class InterpolatedField(Field):

    def __init__(self, field_times: np.ndarray, field_vals: np.ndarray):
        super().__init__()
        if len(field_times) != len(field_vals):
            raise Warning(f"Time array and field array must be of same size! "
                          f"{len(field_times)} ≠ {len(field_vals)}")

        self.field_times = field_times
        self.field_vals  = field_vals

    def value(self, times):
        """
        Evaluate the field at `times` interpolating `field_vals` and `field_times`
        """
        t = np.asarray(times)
        return np.interp(t, self.field_times, self.field_vals)