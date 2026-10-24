import numpy as np

from .base import Potential


class Morse(Potential):
    """
    Morse potential (M) is a 1D potential with a repulsive barrier. 
    The parameters 'D' and 'beta' control the well depth and barrier hardness.
    """

    _serialized_parameters = {"D": "well_depth", "beta": "beta"}

    def __init__(self, D: float, beta: float):
        self.well_depth = D
        self.beta       = beta
        self.name       = "Morse"
        self.acronym    = "Morse"
        super().__init__(name=self.name, acronym=self.acronym)

    def print(self, short: bool = True, latex: bool = False):

        name = self.acronym if short else self.name
        param = fr"$D={self.well_depth:.2f}$, $\beta={self.beta:.2f}$" if latex \
              else f"D = {self.well_depth:.2f}, β = {self.beta:.2f}" 
        return f"{name} {param}"
    
    def value(self, position_array):
        r = np.asarray(position_array)
        
        morse = self.well_depth * (np.exp(-2 * self.beta * r) - 2 * np.exp(-self.beta * r))
        
        return morse
    
    def first_derivative(self, position_array):
        r = np.asarray(position_array)
        
        morse = 2*self.well_depth*self.beta*np.exp( -2*self.beta*r)*( np.exp( self.beta*r ) - 1 )
        
        return morse
    
    def second_derivative(self, position_array):
        r = np.asarray(position_array)
        
        morse = 4*self.well_depth*self.beta**2*np.exp( -2*self.beta*r )*( np.exp( self.beta*r ) - 0.5 )
        
        return morse

    def turning_points(self, energy):

        shape = np.shape(energy)
        energy = np.asarray(energy)

        delta = np.sqrt(energy/self.well_depth + 1)
        
        with np.errstate(invalid='ignore'):           # log(1-delta) is nan on purpose for E>0
            rm = -(1/self.beta)*np.log(1 + delta)
            rM = -(1/self.beta)*np.log(1 - delta)

        rm = np.where(energy < -self.well_depth, np.nan, rm)
        rM = np.where((energy >= 0) | (energy < -self.well_depth), np.nan, rM)

        return np.stack((rm, rM), axis=-1).reshape(shape + (2,))

    def action(self, E: float | np.ndarray):

        return (np.sqrt(2)/self.beta) * \
                   ( np.sqrt(self.well_depth) - np.sqrt(-E) )

    def angular_frequency(self, E):
        return self.beta * np.sqrt(-2*E)

    def frequency(self, E):
        return self.angular_frequency(E)/(2 * np.pi)

    def period(self, E):
        return 1/self.frequency(E)

    def angle(self, E, r):

        # eps vai de 0 (fundo do poço) a 1 (dissociação)
        eps = (E + self.well_depth) / self.well_depth
    
        # Termo X (Cosseno modificado) - Perfeito, não mude nada
        x = (1 - (1 - eps) * np.exp(self.beta * r)) / np.sqrt(eps)
    
        # Termo Y (Seno modificado) - CORRIGIDO
        # p = self.momentum(E, r)
        y = self.momentum(E, r) * np.exp(self.beta * r) * np.sqrt(1 - eps) / np.sqrt(2 * self.well_depth)
    
        return np.arctan2(np.abs(y), x)

    def _batch_action(self, E, M: int = 300):
            E = np.atleast_1d(np.asarray(E, dtype=float))
            bound = E < 0                      # only bound motion has a well-defined action
            result = np.full(E.shape, np.nan)
            if not np.any(bound):
                return result
    
            Eb = E[bound]
            return self.action(Eb)

    def _batch_angle(self, E, r):

        return self.angle(E, r)
        
