"""
    AimetTE -- Trailing-edge noise prediction using Amiet's analytical model

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

class AimetTE:
    def __init__(self, observers:list, data:pd.Series, freq:list = None, \
        phipp_method:str='rozenberg', coh_method:str='crocos', spectra:str=None, coherence:str=None,\
        cinf:float = 343, span:float = 0.5715, chord:float = 0.3048):
        """
        Initialize the AimetTE class.

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
        if freq is None: 
            freq = np.linspace(0, 20000, 100)
            freq[0] = 1e-6 # Avoid zero frequency
        self.freq = freq
        self.observers = observers
        self.phipp_method = phipp_method.lower()
        self.coh_method = coh_method.lower()
        self.data = data
        if self.phipp_method not in ['simulation', 'experiment']:
            self.spectra_path = spectra
        if self.coh_method not in ['simulation', 'experiment']:
            self.coherence_path = coherence
            
        # Checking input data validity
        try:
            check_data(data)
        except ValueError as e:
            raise ValueError(f"Data check failed: {e}")
        
        self.chord = data['chord'] if 'chord' in data else chord
        self.span = data['span'] if 'span' in data else span
        self.cinf = data['cinf'] if 'cinf' in data else cinf
        # Initialize the output variables Directivity and Spp
        self.Spp, self.Directivity  = np.zeros((len(freq), len(observers))), np.zeros((len(freq), len(observers)), dtype=complex)
        
    def compute_spectra(self):
        """
        Computes the spectra and coherence based on the selected methods.
        Raises:
            ValueError: If the spectra or coherence files do not exist when using 'simulation' or 'experiment'.
            ValueError: If the phipp_method or coh_method is not recognized.
        """
        if self.phipp_method == 'simulation' or self.phipp_method == 'experiment':
            if os.path.exists(self.spectra_path):
                freq_phipp, Phipp = self.read_spectra(self.spectra_path)
                print('Frequency and spectra loaded, using user-provided frequency and spectra.')
            else: 
                raise ValueError("Spectra file does not exist.")
        else :
            phipp_model = phipp_models.PhippModels(self.phipp_method, self.data, normalization='frequency', spectral_normalization=2e-5)
            freq_phipp = self.freq
            Phipp = phipp_model.compute_phipp(self.freq)
        
        if self.coh_method == 'simulation' or self.coh_method == 'experiment':
            if os.path.exists(self.coherence_path):
                freq_Ly, Ly = self.read_spectra(self.coherence_path)
                print('Frequency and coherence loaded, using user-provided frequency and coherence.')
            else:
                raise ValueError("Coherence file does not exist.")
        else:
            phipp_model = phipp_models.PhippModels(self.phipp_method, self.data, normalization='frequency', spectral_normalization=2e-5)
            Ly = phipp_model.compute_corrlen(self.freq, bc=0.52, Uc=24)
            freq_Ly = self.freq
        
        if math.isclose(freq_phipp[0], freq_Ly[0]) == False:
            print("Spectra and coherence frequencies do not match.\nInterpolating to match frequencies.")
            Ly_interp = np.interp(freq_phipp, freq_Ly, Ly)
            Ly = Ly_interp
        
        self.freq = freq_phipp
        self.Phipp = Phipp
        self.Ly = Ly

    def calculate_Spp(self):
        """ Main function to calculate the far-field sound pressure spectral density (Spp) based on Aimet's model with leading-edge corrections.
        over the specified frequency range and observer locations.

        Returns:
            Spp (ndarray): Sound pressure spectral density in dB, shape (n_freqs, n_observers).
        """
        Uinf = self.data['Uref']
        Ma = Uinf/self.cinf
        be = 1 - Ma**2
        for i, freq in enumerate(self.freq):
            phippi = self.Phipp[i]
            Lyi = self.Ly[i]
            kc = 2.0 * math.pi * freq/self.cinf*self.chord
            for j, obs in enumerate(self.observers):
                x, y, z = obs[0], obs[1], obs[2]
                SO = np.sqrt(x**2 + be*(y**2 + z**2))
                self.Directivity[i,j] = rdirsupTE(freq, x, y, z, self.chord, Uinf,24, self.cinf)
                Spp = (kc*z/(4.0*math.pi*SO))**2 * (2.0 * self.span) * ((abs(self.Directivity[i, j]))**2) * phippi * Lyi
                Spp = 2.0* math.pi * np.abs(Spp)                    # Convert from Pa^2/(rad/s) to Pa^2/(Hz)
                self.Spp[i, j] = 10 * np.log10(Spp/(2e-5)**2)

        if self.Spp.any()< -25: 
            self.Spp[self.Spp < -25] = -25
        return self.Spp
    
    def compute_Aimet(self):
        """
        Wrapper function to compute the Aimet TE spectra and coherence.
        """
        # Calculate the spectra and coherence to prep for the Aimet model
        try: 
            self.compute_spectra()
            # Calculate the spectra
            Spp = self.calculate_Spp()
            print("Aimet TE spectra calculated successfully.")
        except Exception as e:
            print(f"Error in Aimet TE spectra calculation: {e}")
            raise
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
        print('Loading spectra file:', filename)
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
