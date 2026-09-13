import numpy as np

from .base import Potential


class MorseSoftCoulomb(Potential):
    """
    Morse-soft-Coulomb potential (MsC) is a 1D soft potential with a Morse 
    barrier at the origin. The parameter 'alpha' controls the well depth and
    barrier hardness.
    """
    
    def __init__(self, alpha: float):
        self.alpha      = alpha
        self.well_depth = 1 / alpha
        self.beta       = 1 / (alpha * np.sqrt(2))
        self.name       = "Morse-soft-Coulomb"
        self.acronym    = "MsC"
        super().__init__(name=self.name, acronym=self.acronym)
    
    def value(self, position_array):
        r = np.asarray(position_array)
        
        coulomb = -1 / np.sqrt(r**2 + self.alpha**2)
        morse = self.well_depth * (np.exp(-2 * self.beta * r) - 2 * np.exp(-self.beta * r))
        
        return np.where(r > 0, coulomb, morse)
    
    def first_derivative(self, position_array):
        r = np.asarray(position_array)
        
        coulomb = r*(r**2 + self.alpha**2)**(-3/2)
        morse = 2*self.well_depth*self.beta*np.exp( -2*self.beta*r)*( np.exp( self.beta*r ) - 1 )
        
        return np.where(r > 0, coulomb, morse)
    
    def second_derivative(self, position_array):
        r = np.asarray(position_array)
        
        coulomb = (r**2 + self.alpha**2)**(-3/2) - 3*r**2*(r**2 + self.alpha**2)**(-5/2)
        morse = 4*self.well_depth*self.beta**2*np.exp( -2*self.beta*r )*( np.exp( self.beta*r ) - 0.5 )
        
        return np.where(r > 0, coulomb, morse)
    
    # def return_points(self, energy):
        
    #     if energy < -self.well_depth:
    #         raise Warning(f"Energy must be greater than the well depth ({-self.well_depth})")
    #         return (None, None) 

    #     rm = -self.alpha*np.sqrt(2)*np.log( np.sqrt( self.alpha*energy + 1 ) + 1 )
    #     rM = np.sqrt( 1/(energy**2) - self.alpha**2 )
        
    #     if energy >= 0:
    #         return rm, None
    #     else:
    #         return rm, rM

    def return_points(self, energy):

        shape = np.shape(energy)
        energy = np.asarray(energy)
        rm = -self.alpha*np.sqrt(2)*np.log( np.sqrt( self.alpha*energy + 1 ) + 1 )
        rM = np.sqrt( 1/(energy**2) - self.alpha**2 )

        rm = np.where(energy < -self.well_depth, np.nan, rm)
        rM = np.where((energy >= 0) | (energy < -self.well_depth), np.nan, rM)

        return np.stack((rm, rM), axis=-1).reshape(shape + (2,))
        
