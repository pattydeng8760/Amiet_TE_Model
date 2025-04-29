"""
    Aimet_TE.py: The main module for the Amiet TE model.
"""

import math
import numpy as np
import pandas as pd
from .calc_rdirsupTE import rdirsupTE
#from .calc_directivity import rdirsupTE
from WPS_Model import phipp_models

class AimetTE:
    def __init__(self, freq:list, observers:list, data:pd.DataFrame, phipp_method:str='rozenberg', cinf:float = 343, span:float = 0.5715, chord:float = 0.3048):
        """
        Initialize the AimetTE class.

        Parameters:
        freq (list): Frequency array or scalar (Hz).
        observers (list): List of observer coordinates.
        data (pd): DataFrame containing the input data.
        method (str): Method to be used for calculations.
        kwargs: Additional keyword arguments for specific calculations.
        """
        self.freq = freq
        self.observers = observers
        self.phipp_method = phipp_method
        self.cinf = cinf
        self.span = span
        self.chord = chord
        self.data = data
        # Initialize the output variables Directivity and Spp
        self.Spp, self.Directivity  = np.zeros((len(freq), len(observers))), np.zeros((len(freq), len(observers)), dtype=complex)
        
        # Initialize the phipp modela
        self.phipp_model = phipp_models.PhippModels(phipp_method, data , normalization='frequency', spectral_normalization=2e-5)

    def calculate_Spp(self):
        Uinf = self.data['Uref']
        Ma = Uinf/self.cinf
        be = 1 - Ma**2
        Phipp = self.phipp_model.compute_model(self.freq)
        Ly = self.phipp_model.compute_corrlen(self.freq)
        
        for i, freq in enumerate(self.freq):
            phippi = Phipp[i]
            Lyi = Ly[i]
            kc = 2.0 * math.pi * freq/self.cinf*self.chord
            for j, obs in enumerate(self.observers):
                x, y, z = obs[0], obs[1], obs[2]
                print('Obs: {0:2.2f} {1:2.2f} {2:2.2f}'.format(x, y, z), flush=True)
                SO = np.sqrt(x**2 + be*(y**2 + z**2))
                print('SO: {0:2.2f}'.format(SO), flush=True)
                self.Directivity[i,j] = rdirsupTE(freq, x, y, z, self.chord, Uinf, 1/0.85, self.cinf)
                Spp = (kc*z/(4.0*math.pi*SO))**2 * (2.0 * self.span) * ((abs(self.Directivity[i, j]))**2) * phippi * Lyi
                Spp = 2.0* math.pi * np.abs(Spp)                    # Convert from Pa^2/(rad/s) to Pa^2/(Hz)
                print('Coeff: {0:2.2f}'.format(np.real(kc*z/(4.0*math.pi*SO))**2 ), flush=True)
                self.Spp[i, j] = 10 * np.log10(Spp/(2e-5)**2)
        return self.Spp