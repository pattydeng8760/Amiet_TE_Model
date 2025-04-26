"""
    phipp_Rozenberg -- Aerodynamic wall pressure spectrum Rozenberg model

    Computes the power spectral density (PSD) of wall pressure fluctuationsusing the Rozenberg model 

    :return: Coefficients for the Rozenberg model of wall pressure fluctuations for the standard model
    :rtype: float

    :notes:
        - Input variables must be dimensional (i.e., not non-dimensionalized).
        - The model captures the influence of external/internal time scale ratio,
            wall shear stress, and boundary layer properties.
        - Kinematic viscosity at the wall is computed as: ``nuwall = muwall / rhowall``.
        - Pulsation (angular frequency) is computed as: ``omega = 2π * freq``.

    :references:
        - M. Rozenberg, Phd. thesis 2007-44, ECL, page 133
        - Rozenberg, Yannick, Gilles Robert, and Stéphane Moreau. "Wall-pressure spectral model including the adverse pressure gradient effects." AIAA journal 50.10 (2012): 2168-2179.

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


def phipp_rozenberg(data) -> tuple:
    """ 
    Computes the power spectral density (PSD) of wall pressure fluctuations using the Rozenberg model.
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
    
    
    #Data from 2012 --> AIAAj50-10-2012 Rozenberg
    Ue = data['Ue']
    delta = data['delta']
    delta_star = data['delta_star']
    theta = data['theta']
    beta_c = data['beta_c']
    # Calculate Delta = delta/delta_star
    try:
        Delta = data['Delta']
    except:
        Delta = data['delta']/data['delta_star']
    # Calculate tau_max as max(\mu du/dy) if not provided use tau_w
    try:
        tau_max = data['tau_max']
    except:
        tau_max = data['tau_w']
    # Calculate the Cole's Wake Strength Parameter
    if isinstance(data['PI'],str):
        PI = eval(data['PI'])
        if PI == None:
            PI = 0.8*(beta_c+0.5)**(3/4)
            if np.isnan(PI) or type(PI) == complex:
                PI = 0.8*abs(beta_c+0.5)**(3/4)
    else:
        PI = data['PI']
    
    SS_ = Ue/(tau_max**2*delta_star)
    FS_ = delta_star/Ue
    Rt_ = data['Rt']
    e_ = 3.7+1.5*beta_c
    d_ = 4.76*(1.4/Delta)**(0.75)*(0.375*e_-1)
    if np.isnan(d_) or type(d_) == complex:
        d_ = 4.76*abs(1.4/Delta)**(0.75)*(0.375*e_-1)
    
    a_ = (2.82*Delta**2*(6.13*Delta**(-0.75)+d_)**e_)*(4.2*(PI/Delta)**(0.5)-1)
    if np.isnan(a_) or type(a_) == complex:
        a_ = (2.82*abs(Delta)**2*(6.13*abs(Delta)**(-0.75)+abs(d_))**abs(e_))*(4.2*(abs(PI)/abs(Delta))**(0.5)-1)
    b_ = 2.0
    c_ = 0.75
    f_ = 8.8
    g_ = -0.57
    h_ = np.min((3, 19/np.sqrt(abs(Rt_))))+7
    i_ = 4.76
    
    return (SS_, FS_, Rt_, a_, b_, c_, d_, e_, f_, g_, h_, i_)