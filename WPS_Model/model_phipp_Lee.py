"""
    phipp_Lee -- Aerodynamic wall pressure spectrum Lee's model

    Computes the power spectral density (PSD) of wall pressure fluctuationsusing the Lee's model

    :return: Coefficients for the Lee's model of wall pressure fluctuations for the standard model
    :rtype: float

    :notes:
        - Input variables must be dimensional (i.e., not non-dimensionalized).
        - The model captures the influence of external/internal time scale ratio,
            wall shear stress, and boundary layer properties.
        - Kinematic viscosity at the wall is computed as: ``nuwall = muwall / rhowall``.
        - Pulsation (angular frequency) is computed as: ``omega = 2π * freq``.

    :references:
        - Lee, Seongkyu, et al. “Turbulent Boundary Layer Trailing-Edge
            Noise: Theory, Computation, Experiment, and Application.”
            Progress in Aerospace Sciences 126 (October 1, 2021): 100737.
            https://doi.org/10.1016/j.paerosci.2021.100737.

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

def phipp_Lee(data) -> tuple:
    """
    Computes the power spectral density (PSD) of wall pressure fluctuations using the Lee model.
    Args:
        data (pd.DataFrame): Input data containing parameters: Ue, delta, tau_w, PI, Rt.
        Ue (float): External velocity.
        delta_star (float): Boundary layer displacement thickness.
        delta (float): Boundary layer thickness.
        Delta (float): Ratio of boundary layer thickness to displacement thickness.
        theta (float): Momentum thickness.
        tau_w (float): Wall shear stress.
        tau_max (float): Maximum shear stress, if not provided, use tau_w.
        PI (float): Cole's Wake Strength Parameter, if not provided, use 0.8*(beta_c+0.5)**(3/4).
        beta_c (float): Clauser BL static pressure gradient coefficient.
        Rt (float): Ratio of external to internal time scales.
    Returns:
        (tuple): Coefficients for the Rozenberg model of wall pressure fluctuations for the standard model.
    """

    Ue = data['Ue']
    delta = data['delta']
    delta_star = data['delta_star']
    theta = data['theta']
    beta_c = data['beta_c']
    tau_w = data['tau_w']
    # Calculate Delta = delta/delta_star
    try:
        Delta = data['Delta']
    except:
        Delta = data['delta']/data['delta_star']
    # Calculate the Cole's Wake Strength Parameter
    if isinstance(data['PI'],str):
        PI = eval(data['PI'])
        if PI == None:
            PI = 0.8*(beta_c+0.5)**(3/4)
            if np.isnan(PI) or type(PI) == complex:
                PI = 0.8*abs(beta_c+0.5)**(3/4)
    else:
        PI = data['PI']
    Rt_ = data['Rt']
    # Calculate the coefficients for the Lee model
    e_ = 3.7+1.5*beta_c
    d_star = 4.76*(1.4/Delta)**(0.75)*(0.375*e_-1)
    d_ = max(1,d_star)
    
    a_star = (2.82*Delta**2*(6.13*Delta**(-0.75)+d_)**(e_))*(4.2*(PI/Delta)**(0.5)+1)
    a_ = max(a_star, (0.25*beta_c-0.52)*a_star)
    b_ = 2.0
    c_  = 0.75
    f_  = 8.8
    g_ = -0.57
    h_ = min(3, (0.139 + 3.1043*beta_c)) + 7.0
    i_ = 4.76 
    SS_ = Ue/(tau_w**2*delta_star)
    FS_ = delta_star/Ue
    
    return (SS_, FS_, Rt_, a_, b_, c_, d_, e_, f_, g_, h_, i_)