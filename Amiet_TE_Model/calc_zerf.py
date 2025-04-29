"""
    zerf -- Complex error function

    Computes the complex error function erf(z) for a given complex input ``z``.

    The complex error function is defined as:

    .. math::

        \\mathrm{erf}(z) = \\frac{2}{\\sqrt{\\pi}} \\int_0^z e^{-t^2} \\, dt

    where ``z`` is a complex number.

    The complementary error function is:

    .. math::

        \\mathrm{erfc}(z) = 1 - \\mathrm{erf}(z)

    The algorithm is based on a combination of series expansions and summations for high numerical accuracy
    across the complex plane.

    :param z: Complex input value.
    :type z: complex

    :return: Value of the complex error function erf(z).
    :rtype: complex

    :notes:
        - For purely real inputs, the native real `math.erf` function is used.
        - Special handling is performed if the imaginary part of ``z`` is negative (via conjugation).
        - Summation cutoff is based on machine precision (~1e-15).

    :references:
        - M. Abramowitz and I.A. Stegun, *Handbook of Mathematical Functions*, Dover Publications, 1970, pp. 297–299.

    :author:
        GENEAU Dominic (original Fortran code)  
        DENG Patrick (Python adaptation, modularization, and enhancement)

    :history:
        - 2024-01-15 -- Original file creation.
        - 2025-04-28 -- Python modularization and enhancement for structured documentation.
"""

import numpy as np
import math
from scipy.special import erf

def zerf(z: complex) -> complex:
    return erf(z)
    

# def zerf(z: complex) -> complex:
#     """
#     Computes the complex error function erf(z) for a given complex input.

#     Args:
#         z (complex): Complex number input.

#     Returns:
#         complex: Value of the complex error function erf(z).
#     """
#     if not isinstance(z, complex):
#         z = complex(z)
    
#     x = np.real(z)
#     y = np.imag(z)

#     # Special case: purely real input
#     if y == 0.0:
#         return np.real(math.erf(x))
    
#     # If imaginary part negative, conjugate trick will be used
#     conjugate_flag = False
#     if y < 0.0:
#         conjugate_flag = True
#         y = -y

#     x2 = x * x
#     exp_neg_x2 = math.exp(-x2)

#     # Compute first part
#     e2ixy = np.exp(-2j * x * y)
#     if x == 0.0:
#         part1 = 1j * y / math.pi
#     else:
#         part1 = exp_neg_x2 / (2.0 * math.pi * x) * (1.0 - e2ixy)

#     # Compute second and fourth parts
#     machine_eps = 1e-15
#     n_max = int(np.sqrt(1.0 - 4 * np.log(machine_eps))) + 4

#     Hr2 = 0.0
#     Hi2 = 0.0
#     part2 = 0.0

#     for n in range(1, n_max + 1):
#         n24 = (n * n) / 4.0
#         dd = np.exp(-n24) / (n24 + x2)
#         part2 += dd
#         dd_exp = dd * np.exp(-n * y)
#         Hr2 += dd_exp
#         Hi2 += dd_exp * n

#     part2 = exp_neg_x2 * x * math.pi * part2
#     part4 = -e2ixy * exp_neg_x2 / (2.0 * math.pi) * (Hr2 * x + 1j * (Hi2 / 2.0))

#     # Compute third part
#     m_max = int(2.0 * y)
#     n = max(1, m_max - n_max)
#     m_max = m_max + n_max - n

#     Hr3 = 0.0
#     Hi3 = 0.0
#     for m in range(0, m_max + 1):
#         n24 = (n * n) / 4.0
#         dd = np.exp(n * y - n24) / (n24 + x2)
#         Hr3 += dd
#         Hi3 += dd * n
#         n += 1

#     part3 = -e2ixy * exp_neg_x2 / (2.0 * math.pi) * (Hr3 * x - 1j * (Hi3 / 2.0))

#     # Final summation
#     zerf_out = math.erf(x) + (part1 + part2 + part3 + part4)

#     # Apply conjugate correction if needed
#     if conjugate_flag:
#         zerf_out = np.conjugate(zerf_out)

#     return zerf_out

def zerfc(z: complex) -> complex:
    """
    Computes the complex complementary error function erfc(z) = 1 - erf(z).

    Args:
        z (complex): Complex number input.

    Returns:
        complex: Value of the complementary error function erfc(z).
    """
    return 1.0 - zerf(z)
