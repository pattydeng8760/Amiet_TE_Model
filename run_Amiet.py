# run_amiet.py
"""
Can be used to run the Amiet TE model from a dictionary of arguments.
"""
import argparse
from TEModel import main
from TEModel import parse_args  # assuming parse_args is exposed
import numpy as np

def run_amiet():
    # Set arguments in a dictionary
    args_dict = {
        "output_dir": "./output",
        "output_case": "Test_TE",
        "input_dir": "./input",
        "input_style": "custom",
        "input_data": "",
        "input_data_row": 20,

        # Observer
        "observer_origin": [0.0,0.0,0.0],
        "observer_number": 128,
        "observer_radius": 2.0,
        "selected_freqs": [500, 1000, 2000, 5000],

        # WPS and coherence
        "WPS_model": "experiment",
        "Coherence_model": "rozenberg",
        "WPS_path": 'wall_pressure_spectra_021.dat',
        "Coherence_path": 'correlation_length.dat',

        # Boundary layer parameters
        "Uref": 30.0,
        "Ue": 32.0,
        "delta": 0.01,
        "delta_star": 0.006,
        "theta": 0.005,
        "tau_w": 0.25,
        "beta_c": 14.2,
        "PI": 0.95,
        "Rt": 18.5,
        "dpdx": -1345.0,
        "chord": 0.3048,
        "span": 0.5715,
        "cinf": 343.0, 
        "Uc": 24.0,
    }

    # Convert to Namespace
    args = argparse.Namespace(**args_dict)

    # Run the TE model
    main(args)

if __name__ == "__main__":
    run_amiet()
