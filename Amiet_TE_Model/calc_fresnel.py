"""
    fresnel -- Fresnel integrals

    Computes the Fresnel integral of a complex number ``z``, based on the 
    relationship between the Fresnel integrals and the complex error function (erf).

    The Fresnel integral E(z) is defined as:

    .. math::

        E(z) = \\int_{0}^{z} \\frac{\\exp(i t)}{\\sqrt{2 \\pi t}} \\, dt

    which can be decomposed as:

    .. math::

        E(z) = C_2(z) + i S_2(z)

    where:

    - :math:`C_2(z)` is the cosine component:

      .. math:: C_2(z) = \\int_{0}^{z} \\frac{\\cos(t)}{\\sqrt{2 \\pi t}} \\, dt

    - :math:`S_2(z)` is the sine component:

      .. math:: S_2(z) = \\int_{0}^{z} \\frac{\\sin(t)}{\\sqrt{2 \\pi t}} \\, dt

    The computation is based on the relation between E(z) and the complex error function.

    :param z: Complex input value.
    :type z: complex

    :return: Value of the Fresnel integral E(z).
    :rtype: complex

    :notes:
        - The complex error function `zerf` must be correctly defined and imported.
        - Accurate for complex arguments.
        - Useful in diffraction, scattering, and wave propagation problems.

    :references:
        - M. Abramowitz and I.A. Stegun, *Handbook of Mathematical Functions*, Dover Publications, 1970, pp. 300–301.

    :author:
        GENEAU Dominic (original Fortran code)  
        DENG Patrick (Python adaptation, modularization, and enhancement)

    :history:
        - 2024-01-15 -- Original file creation.
        - 2025-04-28 -- Python enhanced version based on structured modularization.
"""

import numpy as np
from .calc_zerf import zerf

def FresCS2(z) -> complex:
    """
    Computes the Fresnel integral E(z) for a given complex input.

    Args:
        z (complex): Complex number input.

    Returns:
        complex: Value of the Fresnel integral E(z).
    """
    # Validate input
    if not isinstance(z, complex):
        z = complex(z)

    # Argument transformation for complex error function
    arg_erf = (1.0 - 1j) * np.sqrt(z / 2.0)

    # Compute using the complex error function
    ztmp = zerf(arg_erf)

    # Final Fresnel integral computation
    fresnel_out = 0.5 * (1.0 + 1j) * ztmp

    return fresnel_out
