# Wall Pressure Spectrum Models

## Overview

This Python module provides aerodynamic wall-pressure spectral density (WPS) models and spanwise correlation length computations for turbulent boundary layers.  
The following models are implemented:

- **Goody model** (`phipp_Goody`)  
- **Rozenberg model** (`phipp_Rozenberg`)  
- **Lee model** (`phipp_Lee`)  
- **Corcos correlation model** (`corrlen_Corcos`)  
- **Unified model handler** (`PhippModels` class)

These models predict wall pressure spectra and correlation lengths, crucial for aeroacoustic noise predictions, boundary-layer characterization, and hybrid CAA simulations.

---

## Module Structure

| Model | Purpose | Function/Class |
|:---|:---|:---|
| Goody model | Empirical model for surface pressure fluctuations in boundary layers. | `phipp_Goody` |
| Rozenberg model | Extension of Goody model to account for adverse pressure gradients. | `phipp_Rozenberg` |
| Lee model | Further refinement for TBL trailing-edge noise applications. | `phipp_Lee` |
| Corcos model | Computes spanwise correlation length based on frequency and convection velocity. | `corrlen_Corcos` |
| Unified handler | Class to automatically compute WPS for any selected model and correlation length. | `PhippModels` |

---

## Installation

Clone this repository and ensure required packages are installed:

```bash
pip install numpy pandas
```

or install them manually if needed:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Direct Function Use

Example for computing coefficients:

```python
import pandas as pd
from model_phipp_Goody import phipp_Goody
from model_corrlen_Corcos import corrlen_Corcos

# Input data as pandas DataFrame
data = pd.DataFrame({
    'Ue': [50.0],           # External velocity (m/s)
    'delta': [0.05],        # Boundary layer thickness (m)
    'tau_w': [0.3],         # Wall shear stress (Pa)
    'PI': [0.8],            # Wake strength parameter
    'Rt': [500],            # Reynolds number time scale ratio
    'beta_c': [0.2]         # Pressure gradient parameter
})

coefficients = phipp_Goody(data)
print(coefficients)

# Compute correlation length
import numpy as np
freq = np.linspace(0, 1000, 500)
corrlen = corrlen_Corcos(freq, bc=0.52, Uc=0.85)
print(corrlen)
```

---

### 2. Using the Unified `PhippModels` Class

Example to compute PSD spectrum:

```python
from phipp_models import PhippModels

# Initialize model
wps_model = PhippModels(model='rozenberg', data=data, normalization='frequency')

# Compute wall pressure spectrum
freq = np.linspace(0, 1000, 500)
wps_spectrum = wps_model.compute_model(freq)

# Compute correlation length
corr_length = wps_model.compute_corrlen(freq)
```

---

## Models and References

### Goody Model

- Goody, M., "Empirical spectral model of surface pressure fluctuations", *AIAA Journal*, 42(9), 1788–1794, 2004.

### Rozenberg Model

- Rozenberg, Y., Robert, G., Moreau, S., "Wall-pressure spectral model including adverse pressure gradient effects", *AIAA Journal*, 50(10), 2168–2179, 2012.

### Lee Model

- Lee, S., et al., "Turbulent Boundary Layer Trailing-Edge Noise: Theory, Computation, Experiment, and Application", *Progress in Aerospace Sciences*, 126, 100737, 2021.

### Corcos Model

- Corcos, G.M., "Resolution of pressure in turbulence", *J. Acoust. Soc. Am.*, 35(2), 192–199, 1963.
- Corcos, G.M., "The structure of turbulent pressure field in boundary-layer flows", *J. Fluid Mech.*, 18, 353–378, 1964.

---

## Authors

- Lionel GAMET (Original Fortran codes)
- Dominic GENEAU (adapation from Matlab to Python)
- Patrick DENG (Python translation and modularization)

---

## History

- 2009-08-24: Initial Fortran code creation.
- 2009-10-01: Replacement of arguments (rhowall → tauwall).
- 2024-02-16: First Python translation and adaptation.
- 2025-04-16: Modularization, class integration, and cleanup for release.

---