"""
    phipp_Goody -- Aerodynamic wall pressure spectrum Goody model

    Computes the power spectral density (PSD) of wall pressure fluctuations
    using the Goody model, with modifications based on Lee et al. (2021).

    :return: Coefficients for the Goody model of wall pressure fluctuations for the standard model
    :rtype: float

    :notes:
        - Input variables must be dimensional (i.e., not non-dimensionalized).
        - The model captures the influence of external/internal time scale ratio,
            wall shear stress, and boundary layer properties.
        - Kinematic viscosity at the wall is computed as: ``nuwall = muwall / rhowall``.
        - Pulsation (angular frequency) is computed as: ``omega = 2π * freq``.
        - Constants used:
            - a = 3.0
            - b = 2.0
            - c = 0.75
            - d = 0.5
            - e = 3.7
            - f = 1.1
            - g = -0.57
            - h = 7.0

    :references:
        - M. Rozenberg, PhD thesis, 2007-44, ECL, pages 121 and 133.
        - Goody, M., "Empirical spectral model of surface pressure fluctuations",
            AIAA Journal, 42(9), 1788–1794, 2004.
        - Goody, M. & Simpson, R. L., "Surface pressure fluctuations beneath two-
            and three-dimensional turbulent boundary layers", AIAA Journal, 38(10), 1822–1831, 2000.
        - Lee, Seongkyu, et al., "Turbulent Boundary Layer Trailing-Edge Noise:
            Theory, Computation, Experiment, and Application," Progress in Aerospace Sciences, 126, 100737, 2021.
            DOI: https://doi.org/10.1016/j.paerosci.2021.100737

    :author:
        GAMET Lionel 
        GENEAU Dominic
        DENG Patrick

    :history:
        - 2009-08-24 -- Fortran file creation.
        - 2009-10-01 -- Replaced argument rhowall by tauwall (LG).
        - 2024-02-16 -- Python translation and adaptation.
        - 2025-04-16 -- Python modularization and readapation 
"""

import numpy as np
import pandas as pd

def phipp_goody(data:pd.DataFrame) -> tuple:
    """
    Computes the power spectral density (PSD) of wall pressure fluctuations using the Goody model, with modifications based on Lee et al. (2021).
    Args:
        data (pd.DataFrame): Input data containing parameters: Ue, delta, tau_w, PI, Rt.
        Ue (float): External velocity.
        delta (float): Boundary layer thickness.
        tau_w (float): Wall shear stress.
        Rt (float): Ratio of external to internal time scales.
    Returns:
        (tuple): Coefficients for the Goody model of wall pressure fluctuations for the standard model.
    """
    
    
    Ue = data['Ue']
    delta = data['delta']
    beta_c = data['beta_c']
    tau_w = data['tau_w']
    if isinstance(data['PI'],str):
        PI = eval(data['PI'])
        if PI == None:
            PI = 0.8*(beta_c+0.5)**(3/4)
            if np.isnan(PI) or type(PI) == complex:
                PI = 0.8*abs(beta_c+0.5)**(3/4)
    else:
        PI = data['PI']
    SS_ = Ue/(tau_w**2*delta)
    FS_ = delta/Ue
    Rt_ = data['Rt']
    a_ = 3.0
    b_ = 2.0
    c_ = 0.75
    d_ = 0.5
    e_ = 3.7
    f_ = 1.1
    g_ = -0.57
    h_ = 7
    i_ = 1.0
    
    return (SS_, FS_, Rt_, a_, b_, c_, d_, e_, f_, g_, h_, i_)