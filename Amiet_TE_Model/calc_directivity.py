

#Import necessary python libraries
import numpy as np
import math
    
def FresCS2(z):
#Compute complex erf argument

    ArgErf = (1.0e00-1j)*np.sqrt(z/2.0e00)

#Compute complex erf part

    ztmp = zerf(ArgErf)

#Compute final result

    FresCS2_out = 0.5e00*(1.0e00+1j)*ztmp

    return FresCS2_out

#Function call
def zerf(z):
    #Extract real and imaginary parts of input complex number z
    x = np.real(z)
    y = np.imag(z)

#   Call native language real error function for real numbers

    if y == 0.0:
        zerf_out = np.real(math.erf(x))
        return

    x2 = x*x
    e2x2 = math.exp(-x2)

#We assume a positive imaginary part
    DoConjugate = False
    if y < 0.0:
#       Otherwise, computation is made with the conjugate of z
#       The property : erf(conj(z)) = conj( erf(z) ) must then
#       be used at the end of this routine
        DoConjugate = True
        y = -y

#First part of the error function

    e2ixy = math.exp(np.imag(-2.0*x*y))
    if x == 0.0:
        part1 = np.imag(y/math.pi)
    else:
        part1 = e2x2/(2.0*math.pi*x)*(1.0 - e2ixy)

#Second/Fourth parts of the computation
#--------------------------------------

#Compute upper sum index assuming a relative numerical precision equal to 
#machine precision

#   Small number close to machine precision
    ZERONUM = 1e-15
    NMAX = int(np.sqrt(1.0-4*np.log(ZERONUM)))
#   Add a few for more precision
    NMAX = NMAX + 4

#Second and fourth part of error function

    part2 = 0.0
    Hr = 0.0
    Hi = 0.0

    for n in range(1,NMAX+1):
#       Second part
        n24 = (n*n)/4.0
        dd = np.exp(-n24)/(n24+x2)
        part2 = part2 + dd
#       Fourth part
        dd = dd*np.exp(-n*y)
        Hr = Hr + dd
        Hi = Hi + dd*n

    part2 = e2x2*x*math.pi * part2

    part4 = -e2ixy*e2x2/2/math.pi * (Hr*x + 1j*(Hi/2.0))
    
#Third part of the computation
#-----------------------------

#Compute upper sum index

#   The maximum of the exponents for third part is reached roughly at 2y

    MMAX = int(2.0*y)

#   Extend the sums by -/+ NMAX around MMAX
    n = max(1,MMAX-NMAX)
    MMAX = MMAX + NMAX - n

#   Third part of error function
    Hr = 0.0
    Hi = 0.0

    for m in range(0,MMAX+1):
        n24 = (n*n)/4.0
        dd = np.exp(n*y-n24)/(n24+x2)
        Hr = Hr + dd
        Hi = Hi + dd*n
        n = n+1

    part3 = -e2ixy*e2x2/2.0/math.pi * (Hr*x -1j*(Hi/2.0))

#Final summation
#---------------

#Final sum

    zerf_out = math.erf(x) + (part1 + part2 + part3 + part4)

# Case of z having a negative imaginary part

    if DoConjugate:
        zerf_out = np.conjugate(zerf_out)

    return zerf_out

#***************************************************************************
#                      Complimentary error function                 
#***************************************************************************
#

def zerfc(z):
    
    zerfc_out = 1.0 - zerf(z)
    
    return zerfc_out



def rdirsupTE(freq,x1,x2,x3,chord,Uinf,alpha,cinf):
#***************************************************************************
#                        Local variable Definitions                 
#***************************************************************************
#
#   M: Mach number
#   be2: Value of coefficient Beta=1-M**2
#   omega: Pulsation
#
#   k: Acoustic wave number
#   kc: Reduced acoustic wave number
#   Ka: Aerodynamic wave number
#   Kb:  Reduced aerodynamic wave number in direction x1
#   Kby: Reduced aerodynamic wave number in direction x2
#   mu: Corrected reduced aerodynamic wave number
#   epsi: Correction factor
#   S0: Corrected distance
#   [xy]tmp: Real and imaginary parts of a complex number
#
#   kapa: Frequency parameter
#   BB,CC,DD: Coefficients B,C,D as appearing in model formulation
#   Theta1_2: Square of coefficient Theta1 (see model formulation)
#
#   exp4ikapa: Intermediary variable storing exp(+4*i*kapa)
#   HH,GG: Coefficients   H,G as appearing in model formulation
#   L1,L2: Coefficients L1,L2 as appearing in model formulation
#   ztmp: Temporary complex number
#   zfre: Complex number storing the result of Fresnel integral
#   FresCS2: External function computing second Fresnel integral

#***************************************************************************
#                        Routine Instructions                 
#***************************************************************************
#

# Initialization
#----------------

# Mach number

    M = Uinf/cinf
    be2 = 1.0-M*M

# Wave numbers

#   Pulsations
    omega = 2.0*math.pi*freq

#   Acoustic wave number
    k = omega / cinf 

#   Reduced acoustic wave number
    kc = k * chord

#   Aerodynamic wave number
    Ka = omega / Uinf

#   Reduced aerodynamic wave number
    Kb = Ka * chord/2.0

#   Reduced aerodynamic wave number in transverse direction
    Kby = 0.0

#   Corrected reduced aerodynamic wave number
    mu = Kb*M/be2

# Other intermediary variables

#   Frequency parameters
    kapa = np.sqrt(complex((mu*mu-Kby*Kby/be2),0.0))
    exp4ikapa = np.exp(4.0*1j*kapa)

#   Correction factor (used for approximation of imaginary part)
    epsi = 1.0 / np.sqrt( 1.0 + 1.0/(4.0*(mu)))

#   Corrected distance
    S0 = np.sqrt( x1*x1 + be2*(x2*x2+x3*x3))

# Radiation integral computation
#--------------------------------

# Compute terms B, C and D

    BB = alpha*Kb + mu*M + kapa
    CC = alpha*Kb - mu*(x1/S0-M)
    DD = kapa - mu*x1/S0

# Compute square of intermediary variable Theta1
    Theta1_2 = BB / (Kb + mu*M + kapa)

# Compute term H

    HH = (1.0+1j)*(1.0-Theta1_2)*np.conj(exp4ikapa)/ \
        (2.0*np.sqrt(BB*math.pi)*(alpha-1.0)*Kb)

# Compute term G

    GG = (1.0+epsi)*np.exp(1j*(DD+2.0*kapa))*\
            np.sin(DD-2.0*kapa)/(DD-2.0*kapa) + \
        (1.0-epsi)*np.exp(1j*(DD-2.0*kapa))* \
            np.sin(DD+2.0*kapa)/(DD+2.0*kapa)

    zfre = FresCS2(4.0*kapa)
    ztmp = (1.0+1j)*np.conj(exp4ikapa)*zfre
    GG = GG + (1.0+epsi)*0.5*np.conj(ztmp)/(DD-2.0*kapa) - \
        (1.0-epsi)*0.5*ztmp /(DD+2.0*kapa)

    ztmp = (1.0-epsi)*(1.0+1j)/(DD+2.0*kapa) - \
        (1.0+epsi)*(1.0-1j)/(DD-2.0*kapa)
    GG = GG + 0.5*np.exp(2.0*1j*DD)*np.sqrt(2.0*kapa/DD)*\
        np.conj(FresCS2(2.0*DD))*ztmp

# Compute final integral L2 (Leading edge contribution)

    ztmp = exp4ikapa*(1.0-(1.0+1j)*np.conj(zfre))
#   Multiply imaginary part by epsi
    xtmp = np.real(ztmp)
    ytmp = np.imag(ztmp)*epsi
    ztmp = complex(xtmp,ytmp)

#   Compute final integral L2
    L2 = HH*( ztmp - np.exp(2.0*1j*DD) + 1j*(DD+Kb+mu*M-kapa)*GG)

#   Compute final integral L1 (Trailing edge correction)

    zfre = np.conj(FresCS2(2.0*BB))
    L1 = 1.0 - (1.0+1j)*zfre

    if ( (BB-CC) == complex(0.0,0.0) ):
        L1 = L1 + (1.0+1j)*np.exp(-2.0*1j*CC)*np.sqrt(BB)*2.0/np.sqrt(math.pi)
    else:
        zfre = np.conj( FresCS2(2.0*(BB-CC)) )
        L1 = L1 + (1.0+1j)*np.exp(-2.0*1j*CC)*np.sqrt(BB/(BB-CC))*zfre

    L1 = L1 * 1j*np.exp(2.0*1j*CC)/CC

# Radiation integral computation
#--------------------------------
    #breakpoint()
    rdirsupTE_out = L1 + L2

    return rdirsupTE_out

