"""
    gammaln -- Natural logarithm of the real gamma function

    Computes the natural logarithm of the Gamma Euler function for a given real input ``x``
    using an improved Lanczos approximation with 15 terms.

    The gamma function, Γ(x), is mathematically defined as:

    .. math::

        \\Gamma(x) = \\int_{0}^{+\\infty} t^{x-1} e^{-t} \\,dt
    
    We can apply the Lanczos approximation to compute the logarithm of the gamma function.
    .. math::
        \\ln(\\Gamma(x)) = \\ln(\\sqrt{2 \pi}) + \\sum_{k=0}^{m} \\frac{a_k}{x+k} - x + (x-0.5)\\ln(x+g)

    where :math:`a_k` are coefficients of the Lanczos approximation and :math:`g` is a constant.

    This routine returns the natural logarithm of Γ(x) with high numerical accuracy.

    :param x: Real number input (must be positive).
    :type x: float

    :return: Natural logarithm of the Gamma function evaluated at x.
    :rtype: float

    :notes:
        - The input ``x`` must be positive.
        - Uses a 15-term Lanczos approximation for improved numerical stability.
        - Computation is accurate up to machine precision for most practical ranges.

    :references:
        - M. Abramowitz and I.A. Stegun, *Handbook of Mathematical Functions*, Dover Publications, 1970.
        - Press, W.H. et al., *Numerical Recipes: The Art of Scientific Computing*, Cambridge University Press.
        - Lanczos, C., "A precision approximation of the gamma function", J. SIAM Numerical Analysis Series B, 1 (1964).

    :author:
        GAMET Lionel (original Fortran code)
        GENEAU Dominic (Python translation)
        DENG Patrick (Python adaptation, modularization, and enhancement to higher precision)

    :history:
        - 2024-01-15 -- Original file creation.
        - 2025-04-28 -- Python enhanced version with 15-term Lanczos approximation.
"""
import numpy as np
import math

def gammaln(x: float, method:str = None) -> float:
    """
    Computes the natural logarithm of the Gamma function for a given positive input x based on the Lanczos approximation.

    Args:
        x (float): Input value (must be positive).
        method (str, optional): Method to use for computation. Options are 'fast', 'high', or None.
            - 'fast': Fast approximation using a 7-term Lanczos approximation.
            - 'high': High precision using a 15-term Lanczos approximation.
            - None: Uses scipy's gammaln function for better numerical stability but requires the scipy library.
    Raises:
        ValueError: If x is not positive.

    Returns:
        float: the natural logarithm of the Gamma function evaluated at x.
    """
    # Ensure that the input is a positive number 
    if x <= 0.0:
        raise ValueError("Input x must be positive.")
        
    if method == 'fast':
        # Coefficients for the Lanczos approximation
        coef = [
            1.000000000190015e+00, 7.618009172947146e+01,
            -8.650532032941677e+01, 2.401409824083091e+01,
            -1.231739572450155e+00, 1.208650973866179e-03,
            -5.395239384953000e-06
        ]
        g = 5.5
        series = coef[0]
        y = x
        for c in coef[1:]:
            y += 1.0
            series += c / y
        tmp = x + g
        gammaln_out = (x + 0.5) * math.log(tmp) - tmp + math.log(math.sqrt(2 * math.pi) * series / x)

        
    elif method == 'high':
        coef = [
            0.99999999999980993, 676.5203681218851, -1259.1392167224028,
            771.32342877765313, -176.61502916214059, 12.507343278686905,
            -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
            -2.7557313707070068e-9, 2.480158728947673e-10, -2.7557319223985893e-12,
            2.0876756987868098e-13, -1.1470745597729725e-15, 3.0872325690161147e-17 ]
        series = coef[0]
        g = 7.0
        y = x
        for c in coef[1:]:
            y += 1.0
            series += c / y
        tmp = x + g + 0.5
        gammaln_out = (x + 0.5) * math.log(tmp) - tmp + math.log(math.sqrt(2 * math.pi) * series / x)

    else:
        from scipy.special import gammaln as scipy_gammaln
        # Use scipy's gammaln function for better numerical stability
        gammaln_out = scipy_gammaln(x)
    
    return gammaln_out
