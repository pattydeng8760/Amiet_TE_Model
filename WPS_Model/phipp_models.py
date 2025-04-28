import os
import sys
import numpy as np
import pandas as pd
from .model_phipp_Goody import phipp_Goody
from .model_phipp_Rozenberg import phipp_Rozenberg
from .model_phipp_Lee import phipp_Lee
from .model_corrlen_Corcos import corrlen_Corcos
class PhippModels:
    """
    The PhippModels class computes the power spectral density (PSD) of wall pressure fluctuations using the specified model
    
    Args: 
        model (str): The model to use
        data (pd.DataFrame): Input data containing the flow parameters for the model.
        normalization (str): The normalization method to use, either 'frequency', 'struhal', or 'omega'.
        spectral_normalization (str): The spectral normalization method to use to scale the PSD. default is 2e-5.
    
    """
    def __init__(self, model:str, data:pd.DataFrame, normalization:str='frequency', spectral_normalization:float=2e-5):
        assert model.lower() in ['goody','rozenberg', 'lee'], "Model must be 'goody' or 'rozenberg' or 'lee'"
        self.model = model
        self.spectral_norm = spectral_normalization
        if normalization.lower() not in ['frequency','struhal','omega']:
            raise ValueError("Normalization must be 'frequency' or 'struhal' or 'omega'")
        
        # Initialize the model parameters
        self.map_model(data)

    def universal_model(self,freq):
        omega = 2*np.pi*freq
        phipp = (1/self.SS_)*(self.a_*((omega*self.FS_)**self.b_) /( (self.i_*((omega*self.FS_)**self.c_) + self.d_)**self.e_ + ((self.f_*abs(self.Rt_)**self.g_)*(omega*self.FS_))**self.h_ ))
        if np.isnan(phipp).all():
            print("Warning: phipp is NaN for all frequencies.")
            phipp = abs(1/self.SS_)*(self.a_*(abs(omega*self.FS_)**self.b_) /( abs(self.i_*(abs(omega*self.FS_)**self.c_) + self.d_)**self.e_ + ((self.f_*abs(self.Rt_)**self.g_)*abs(omega*self.FS_))**self.h_ ))
        return phipp, omega
    

    def map_model(self,data):
        # Comptue the model coefficients for the universal model
        if self.model.lower() == 'goody':
            inputs = phipp_Goody(data)
        elif self.model.lower() == 'rozenberg':
            inputs = phipp_Rozenberg(data)
        elif self.model.lower() == 'lee':
            inputs = phipp_Lee(data)
            
        # Unpack the inputs and assign them to the class attributes
        (
            self.SS_, self.FS_, self.Rt_,
            self.a_, self.b_, self.c_, self.d_,
            self.e_, self.f_, self.g_, self.h_, self.i_
        ) = inputs

    def compute_model(self,freq,spectral_normalization=None):
        phi_pp, omega = self.universal_model(freq)
        # Normalize the model as (2e-5)**2 or a user defined value float 
        norm_val = (self.spectral_norm)**2 if spectral_normalization is None else spectral_normalization
        # Convert omega to frequency 
        phi_pp *= 2*np.pi #convert to frequency instead of omega phi_pp(omega)*2pi = tilde(phi_pp(f))
        
        if omega[0] == 0:
            phippaux = 10*np.log10(phi_pp[1:]/norm_val)
            phipp = np.concatenate(([phippaux[0]],phippaux), axis=0)
        else:
            phipp = 10*np.log10(phi_pp/norm_val)
        return phipp
    
    def compute_corrlen(self,freq, bc:float=0.52, Uc:float=0.85):
        """
        Computes the spanwise correlation length
        """
        corrlen = corrlen_Corcos(freq, bc, Uc)
        return corrlen
        
