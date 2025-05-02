# README – Amiet Trailing-Edge Noise Model (with Roger & Moreau 2005 LE-TE correction)

## Overview

This repository implements **Amiet’s trailing-edge aeroacoustic model** and augments it with the **leading-edge–trailing-edge (LE-TE) coherent-scattering correction** introduced by Roger & Moreau (2005).  
The wrapper script `ComputeAmiet_TE.py` performs the following:

1. Parses command-line arguments  
2. Loads boundary-layer and wall-pressure data  
3. Generates a ring of observers  
4. Invokes the solver class `AmietTE` (in `Amiet_TE_Model/`)  
5. Writes spectra (`.dat`) and high-resolution plots (`.png`) to user-specified folders  

---

## Installation

```bash
git clone https://github.com/YourOrg/Amiet_TE_Model.git
cd Amiet_TE_Model
python -m pip install -r requirements.txt
```

---

## Quick Start

```bash
python TEModel.py \
    --input_dir   ./input \
    --input_data  TA10_BLparams_zones.csv \
    --input_data_row 10 \
    --output_dir  ./output \
    --output_case TE_Example \
    --observer_number 24 \
    --observer_radius 1.50 \
    --selected_freqs 500 1000 2000 \
    --WPS_model  rozenberg \
    --Coherence_model corcos
```

---

## Command-Line Arguments (Excerpt)

| Flag                          | Default                      | Description |
|-------------------------------|------------------------------|-------------|
| `-o`, `--output_dir`          | `./output`                   | Output folder |
| `-oc`, `--output_case`        | `Aimiet_TE`                  | Prefix for output files |
| `-i`, `--input_dir`           | `./input`                    | Input folder |
| `-d`, `--input_data`          | `TA10_BLparams_zones.csv`    | Boundary-layer parameter file |
| `-dr`, `--input_data_row`     | `10`                         | Row index to extract from CSV |
| `-ob_n`, `--observer_number`  | `12`                         | Number of observers |
| `-ob_r`, `--observer_radius`  | `2.0`                        | Radius of observer ring [m] |
| `-sf`, `--selected_freqs`     | `500 1000 2000`              | Frequencies for directivity plot [Hz] |
| `-wps`, `--WPS_model`         | `rozenberg`                  | WPS model: `goody`, `rozenberg`, `lee`, `experiment`, `simulation` |
| `-coh`, `--Coherence_model`   | `corcos`                     | Coherence model: `corcos`, `experiment`, `simulation` |
| **Physical parameters**       | (see code)                   | Includes `Ue`, `delta`, `theta`, `chord`, `span`, `cinf`, etc. |

For a complete list of arguments, run:

```bash
python ComputeAmiet_TE.py --help
```

---

## Output Structure

```
output/
├── TE_Example_Spectra/
│   ├── Amiet_TE_Spectra_Probe_000.dat
│   ├── Amiet_TE_Spectra_Probe_000.png
│   └── WPS_Spectra.dat
└── TE_Example_Directivity/
    └── Directivity_Patterns_500_1000_2000.png
```

- `.dat` — ASCII table with frequency [Hz] and PSD [dB/Hz]  
- `.png` — 300 dpi figures for spectra and polar directivity plots

---

## Theory

### Amiet’s Trailing-Edge Model

For large Helmholtz numbers \( k c \gg 1 \), the far-field sound pressure spectrum is:

$$
S_{pp}(f, \phi, \theta) = \left( \frac{k_c z}{4\pi R} \right)^2 \cdot 2L \cdot |D(\phi, \theta)|^2 \cdot \Phi_{pp}\left(k_1 = \frac{\omega}{U_c}\right)
$$

Where:
- \( k_c = \omega / U_c \): chordwise gust wavenumber  
- \( z \): distance from mid-chord to trailing edge  
- \( R \): observer distance  
- \( L \): span (acoustically compact assumption)  
- \( D(\phi, \theta) \): directivity function  
- \( \Phi_{pp}(k_1) \): wall-pressure spectrum

### LE-TE Coherent-Scattering Correction (Roger & Moreau 2005)

A correction factor accounts for the interference between leading and trailing edge radiation:

$$
S_{pp}^{LE-TE} = \left|1 - e^{-i k_c c}\right|^2 \cdot S_{pp}(f, \phi, \theta)
$$

This reduces low-frequency over-prediction and reverts to the original Amiet model as \( k_c c \to \infty \).

---

## References

1. Amiet, R. K. (1976). "Noise due to turbulent flow past a trailing edge." *Journal of Sound and Vibration*, 47(3), 387–393.  
2. Roger, M., & Moreau, S. (2005). "Broadband noise from lifted turbulent airfoil trailing edges." *AIAA Journal*, 43(2), 249–260.  
3. Rozenberg, Y., Robert, G., & Moreau, S. (2010). "Wall-pressure spectral density modeling for airfoil broadband trailing-edge noise prediction." *AIAA Journal*, 48(6), 1191–1206.

---

## License

Released under the MIT License. See the `LICENSE` file for details.
