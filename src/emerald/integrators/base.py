"""
Base module with Integrator protocol.

 - Integrator: the general interface for integrating dynamics

"""

from abc import ABC, abstractmethod


class Integrator(ABC):

    @abstractmethod
    def propagate(self, dynamics, states0, t_span, t_eval): ...