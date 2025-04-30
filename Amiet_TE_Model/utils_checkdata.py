import pandas as pd


def check_data(data):
    """
    Check if the data is valid.
    
    Parameters:
        data (pd): DataFrame containing the input data.
    
    """
    print('\nChecking data...')
    if not isinstance(data, pd.Series):
        raise ValueError("Data must be a pandas Series.")
    if 'Ue' not in data:
        raise ValueError("Data must contain 'Ue' key. (edge Velocity)")
    if 'Uref' not in data:
        raise ValueError("Data must contain 'Uref' key. (free stream/reference velocity)")
    if 'delta' not in data:
        raise ValueError("Data must contain 'delta' key. (99 percent boundary layer thickness)")
    if 'tau_w' not in data:
        raise ValueError("Data must contain 'tau_w' key. (wall shear stress)")
    if 'beta_c' not in data:
        raise ValueError("Data must contain 'beta_c' key. (clauser parameter = dp/dx * (theta / tau_w)")
    if 'PI' not in data:
        raise ValueError("Data must contain 'PI' key. (pressure gradient parameter = 0.8*(beta_c+0.5)**(0.75))")
    if 'Rt' not in data:
        raise ValueError("Data must contain 'Rt' key. (ratio of external to internal time scales = delta * u_tau^2/(Ue*nu))")
    if 'delta_star' not in data:
        raise ValueError("Data must contain 'delta_star' key. (boundary layer displacement thickness)")
    if 'theta' not in data:
        raise ValueError("Data must contain 'theta' key. (momentum thickness)")
    if 'dpdx' not in data:
        raise ValueError("Data must contain 'dpdx' key. (pressure gradient)")
    print('Data check complete, no errors in the input.\n')