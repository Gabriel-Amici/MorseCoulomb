"""
Sinusoidal fields implementation
"""

import numpy as np

from .base import Field


class SinusoidField(Field):
    """
    A sinusoidal field of form F(t) = A * sin(ωt + φ), where A is the 
    amplitude, ω is the angular frequency, and φ is the phase.
    """
    def __init__(self, amplitude: float, ang_frequency: float, phase: float):
        super().__init__()
        self.amplitude     = amplitude
        self.ang_frequency = ang_frequency
        self.phase         = phase

    def value(self, times):
        return self.amplitude*np.sin( self.ang_frequency*times + self.phase )