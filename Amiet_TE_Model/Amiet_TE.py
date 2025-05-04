"""
    AmietTE -- Trailing-edge noise prediction using Amiet's analytical model

    Implements Amiet's trailing-edge noise model with extensions to account for 
    coherence effects and empirical/analytical wall-pressure spectrum models. This class 
    calculates the far-field sound pressure spectral density (PSD) and radiation directivity 
    due to turbulent boundary layer–trailing edge interaction for a finite-span airfoil.

    The total acoustic power spectral density at a given observer location is computed as:

    .. math::

        S_{pp}(f, \mathbf{x}) = \\left( \\frac{k_c z}{4 \\pi S_0} \\right)^2 (2 L_y) \\left| D(f, \\mathbf{x}) \\right|^2 \\, \\Phi_{pp}(f) \\, L_y(f)

    where:
        - :math:`k_c` is the chordwise acoustic wavenumber,
        - :math:`S_0` is the retarded distance between the airfoil and observer,
        - :math:`D(f, \\mathbf{x})` is the complex-valued directivity,
        - :math:`\\Phi_{pp}(f)` is the wall-pressure spectrum (modeled or from data),
        - :math:`L_y(f)` is the spanwise coherence length.

    The model includes:
        - User-selectable wall-pressure spectrum models (Goody, Rozenberg, Lee, or file-based),
        - Coherence models (Crocos, experimental, or user-defined),
        - Support for finite span corrections,
        - Frequency interpolation when required.

    :param observers: List of 3D observer coordinates [[x1, x2, x3], ...].
    :type observers: list
    :param data: One-row pandas Series containing boundary layer parameters (e.g., Uref, delta, tau_w).
    :type data: pandas.Series
    :param freq: Optional frequency array (Hz). If None, defaults to linspace from 0–20 kHz.
    :type freq: list or ndarray
    :param phipp_method: Method to compute wall-pressure spectrum. Options: ['rozenberg', 'goody', 'lee', 'simulation', 'experiment'].
    :type phipp_method: str
    :param coh_method: Method to compute spanwise coherence. Options: ['crocos', 'simulation', 'experiment'].
    :type coh_method: str
    :param spectra: Path to .csv or .dat file containing frequency and wall-pressure spectrum data (if using 'experiment' or 'simulation').
    :type spectra: str or None
    :param coherence: Path to file with coherence length data (if using 'experiment' or 'simulation').
    :type coherence: str or None
    :param cinf: Speed of sound (m/s). Default is 343 m/s.
    :type cinf: float
    :param span: Airfoil span (m). Default is 0.5715 m.
    :type span: float
    :param chord: Airfoil chord (m). Default is 0.3048 m.
    :type chord: float

    :attributes:
        - freq (ndarray): Frequency array used for computation.
        - Spp (ndarray): PSD results in dB, shape (n_freqs, n_observers).
        - Directivity (ndarray): Complex-valued directivity matrix (same shape).
        - Phipp (ndarray): Wall-pressure spectrum evaluated at frequencies.
        - Ly (ndarray): Coherence lengths across frequency.

    :raises:
        - ValueError: If required files do not exist or if input validation fails.

    :references:
        - Amiet, R.K., “Noise due to turbulent flow past a trailing edge,” J. Sound Vib., 47(3), 387–393, 1976.
        - Rozenberg, M., “PhD Thesis 2007-44,” École Centrale de Lyon.
        - Roger, M. and Moreau, S., “Back-scattering correction and further extensions of Amiet's trailing-edge noise model,” JSV, 2005–2009.
        - Goody, M. “Empirical spectral model of surface pressure fluctuations,” AIAA J, 42(9), 1788–1794, 2004.

    :author:
        GENEAU Dominic (original implementation)  
        GAMET Lionel (Fortran extensions)  
        DENG Patrick (Python adaptation, spectral model integration, vectorization)

    :history:
        - 2009–2010 -- Original FORTRAN implementation.
        - 2024-01-15 -- Initial Python rewrite.
        - 2025-04-28 -- Modularization, model integration, and documentation refinement.
"""

import math
import os
import numpy as np
import pandas as pd
from .calc_rdirsupTE import rdirsupTE
from .utils_checkdata import check_data
from WPS_Model import phipp_models

class AmietTE:
    def __init__(self, observers:list, data:pd.Series, freq:list = None, \
        phipp_method:str='rozenberg', coh_method:str='crocos', spectra:str=None, coherence:str=None,\
        cinf:float = 343, span:float = 0.5715, chord:float = 0.3048, Uc:float = 24):
        """
        Initialize the AmietTE class.

        Parameters:
            observers (list): List of observer coordinates.
            data (pd): DataFrame containing the input data.
            freq (list): Frequency array or scalar (Hz).
            phipp_method (str): Method to be used for phipp calculations.
            coh_method (str): Method to be used for coherence calculations.
            spectra (str): Path to the spectra file, only used if phipp_method is not 'simulation' or 'experiment'.
            coherence (str): Path to the coherence file, only used if coh_method is not 'simulation' or 'experiment'.
            cinf (float): Free-stream speed of sound (m/s), default is 343 m/s.
            span (float): Span of the airfoil (m), default is 0.5715 m.
            chord (float): Airfoil chord (m), default is 0.3048 m.
        
        """
        print(f"\n{'Computing Amiets Trailing Edge Model':=^100}\n")  
        if freq is None: 
            freq = np.logspace(0, 4.5, num=100, base=10.0) # Default frequency range from 1 Hz to 20 kHz
        self.freq = freq
        self.observers = observers
        self.phipp_method = phipp_method.lower()
        self.coh_method = coh_method.lower()
        self.data = data
        self.spectra_path = spectra if self.phipp_method in ['simulation', 'experiment'] else None
        self.coherence_path = coherence if self.coh_method in ['simulation', 'experiment'] else None
        
        print('Model Settings:')
        print(f"    Frequency range: {freq[0]:.2f} Hz to {freq[-1]:.2f} Hz")
        print(f"    Chord: {chord:.4f} m")
        print(f"    Span: {span:.4f} m")
        print(f"    Speed of sound: {cinf:.2f} m/s")
        print(f"    Wall-pressure spectrum method: {phipp_method}")
        print(f"    Coherence method: {coh_method}")
        print(f"    Spectra file: {spectra}") if self.phipp_method in ['simulation', 'experiment'] else None
        print(f"    Coherence file: {coherence}") if self.coh_method in ['simulation', 'experiment'] else None
            

            
        # Checking input data validity
        try:
            check_data(data)
        except ValueError as e:
            raise ValueError(f"Data check failed: {e}")
        
        self.chord = data['chord'] if 'chord' in data else chord
        self.span = data['span'] if 'span' in data else span
        self.cinf = data['cinf'] if 'cinf' in data else cinf
        self.Uc = data['Uc'] if 'Uc' in data else Uc
        
    def compute_spectra(self):
        """
        Computes the spectra and coherence based on the selected methods.
        Raises:
            ValueError: If the spectra or coherence files do not exist when using 'simulation' or 'experiment'.
            ValueError: If the phipp_method or coh_method is not recognized.
        """
        if self.spectra_path is None or self.coherence_path is None:
            phipp_model = phipp_models.PhippModels(self.phipp_method, self.data, normalization='frequency', spectral_normalization=2e-5, verbose=True)
        
        print(f"\n{'Extracting phipp and coherence':.^60}\n")  
        if self.phipp_method == 'simulation' or self.phipp_method == 'experiment':
            print('----> Extracting phipp from {0:s} data:'.format(self.phipp_method))
            if os.path.exists(self.spectra_path):
                freq_phipp, Phipp = self.read_spectra(self.spectra_path)
                print('    Frequency and spectra loaded, using user-provided frequency and spectra.')
            else: 
                raise ValueError("Spectra file does not exist.")
        else :
            print('----> Extracting phipp from {0:s} model'.format(self.phipp_method))
            freq_phipp = self.freq
            Phipp = phipp_model.compute_phipp(self.freq)
        
        if self.coh_method == 'simulation' or self.coh_method == 'experiment':
            print('----> Extracting coherence from {0:s} data'.format(self.coh_method))
            if os.path.exists(self.coherence_path):
                freq_Ly, Ly = self.read_spectra(self.coherence_path)
                print('    Frequency and coherence loaded, using user-provided frequency and coherence.')
            else:
                raise ValueError("Coherence file does not exist.")
        else:
            print('----> Extracting coherence from {0:s} method'.format(self.coh_method))
            Ly = phipp_model.compute_corrlen(self.freq, bc=0.52, Uc=24)
            freq_Ly = self.freq
        
        if math.isclose(freq_phipp[0], freq_Ly[0]) == False:
            print("----> Spectra and coherence frequencies do not match.\n    Interpolating to match frequencies.")
            Ly_interp = np.interp(freq_phipp, freq_Ly, Ly)
            Ly = Ly_interp
        
        self.freq = freq_phipp
        if math.isclose(self.freq[0],0) == True:
            self.freq[0] = 1e-3 
        self.Phipp = Phipp
        self.Ly = Ly
        print(f"\n{'phipp and coherence extraction complete':.^60}\n")  

    def calculate_Spp(self):
        """ Main function to calculate the far-field sound pressure spectral density (Spp) based on Amiet's model with leading-edge corrections.
        over the specified frequency range and observer locations.

        Returns:
            Spp (ndarray): Sound pressure spectral density in dB, shape (n_freqs, n_observers).
        """
        # Initialize the output variables Directivity and Spp
        print(f"\n{'Calculating Spp':.^60}\n")
        self.Spp, self.Directivity  = np.zeros((len(self.freq), len(self.observers))), np.zeros((len(self.freq), len(self.observers)), dtype=complex)
        Uinf = self.data['Uref']
        Ma = Uinf/self.cinf
        be = 1 - Ma**2
        print('----> Looping over {0:3d} frequencies and {1:3d} observers:'.format(len(self.freq), len(self.observers)))
        for i, freq in enumerate(self.freq):
            phippi = self.Phipp[i]
            Lyi = self.Ly[i]
            kc = 2.0 * math.pi * freq/self.cinf*self.chord
            for j, obs in enumerate(self.observers):
                x, y, z = obs[0], obs[1], obs[2]
                SO = np.sqrt(x**2 + be*(y**2 + z**2))
                self.Directivity[i,j] = rdirsupTE(freq, x, y, z, self.chord, Uinf, self.Uc, self.cinf)
                Spp_tmp = (kc*z/(4.0*math.pi*SO))**2 * (2.0 * self.span) * ((abs(self.Directivity[i, j]))**2) * phippi * Lyi
                Spp = 2.0* math.pi * np.abs(Spp_tmp)                    # Convert from Pa^2/(rad/s) to Pa^2/(Hz)
                # if math.isclose(Spp, 0):            # Avoid log(0), case Spp to 1e-8 such that log10(1e-8/4e-10) = -25
                #     Spp = (3e-5)**2
                self.Spp[i, j] = 10 * np.log10(Spp/(2e-5)**2)

        if self.Spp.any()< -25: 
            self.Spp[self.Spp < -25] = 1e-3
        print(f"\n{'Complete Spp Calculation':.^60}\n")
        return self.Spp
    
    def compute_Amiet(self):
        """
        Wrapper function to compute the Amiet TE spectra and coherence.
        """
        # Calculate the spectra and coherence to prep for the Amiet model
        try: 
            self.compute_spectra()
            # Calculate the spectra
            Spp = self.calculate_Spp()
            print("Amiet TE spectra calculated successfully.")
        except Exception as e:
            print(f"Error in Amiet TE spectra calculation: {e}")
            raise
        print(f"\n{'Complete Amiets Trailing Edge Model':=^100}\n")  
        return Spp
        
    @staticmethod
    def read_spectra(filename, has_header:bool=True):
        """
        Auxiliary function that Reads a two-column file (CSV or .dat) with unknown delimiter,
        returns two numpy arrays (col0, col1).
        
        Parameters:
        filename   : path to file
        has_header : True if first line is header, False otherwise
        
        Returns:
        col0: numpy array of first column, should be frequency 
        col1: numpy array of second column, should be spectral content
        """
        header_arg = 0 if has_header else None
        df = pd.read_csv(
            filename,
            sep=r'[\s,]+',    # any run of whitespace or commas as delimiter
            engine='python',
            header=header_arg
        )
        if df.shape[1] < 2:
            raise ValueError("File must contain at least two columns.")
        # take the first two columns regardless of their names
        arr = df.iloc[:, :2].to_numpy()
        return arr[:, 0], arr[:, 1]
