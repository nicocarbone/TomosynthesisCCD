#%%

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import interpolate
from scipy.special import iv as Bi #2 arguments: order, z
from scipy.special import kv as Bk #2 arguments: order, z
from scipy.special import lpmv as P #3 arguments: order, degree, x
from scipy.special import legendre

def D (mups):
    return 1/(3*mups)

def Dcomp (mua, mups):
    return 1/(3*(mua+mups))

def rs (x,y,z):
    return np.sqrt(x**2+y**2+z**2)

def K (mua, mups):
    return np.sqrt(mua/D(mups))

def distance (x0, y0, z0, x, y, z):
    return np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)

def Green (muao, mupso, x0, y0, z0, x, y, z):
    return (0.75*mupso/math.pi)*np.exp(-K(muao,mupso)*distance(x0, y0, z0, x, y, z))/distance(x0, y0, z0, x, y, z)

def costeta (x0, y0, z0, x, y, z):
    return (x*x0 + y*y0 + z*z0) / (rs(x,y,z) * rs(x0,y0,z0))

def Bm (m, muao, muai, mupso, mupsi, a, x0,y0,z0):
    return -.7957747154e-1*Bk(m+.5, K(muao,mupso) * rs(x0,y0,z0))*(
                                                      -2.*D(mupsi)*a*Bi(m+1.5,K(muai,mupsi)*a)*K(muai,mupsi)*Bi(m+.5,K(muao,mupso)*a)* m 
                                                      -1.* D(mupsi)*a*Bi(m+1.5,K(muai,mupsi)*a)*K(muai,mupsi)*Bi(m+.5,K(muao,mupso)*a)
                                                      -2.*D(mupsi)*Bi(m+.5,K(muai,mupsi)*a)*m**2*Bi(m+.5,K(muao,mupso)*a)
                                                      -1.*D(mupsi)*Bi(m+.5,K(muai,mupsi)*a)*m*Bi(m+.5,K(muao,mupso)*a)
                                                      +2.*a*m*Bi(m+1.5,K(muao,mupso)*a)*K(muao,mupso)*Bi(m+.5,K(muai,mupsi)*a)*D(mupso)
                                                      +2.*Bi(m+.5,K(muao,mupso)*a)*m**2*Bi(m+.5,K(muai,mupsi)*a)*D(mupso)
                                                      +Bi(m+.5,K(muao,mupso)*a)*m*Bi(m+.5,K(muai,mupsi)*a)*D(mupso)
                                                      +a*Bi(m+1.5,K(muao,mupso)*a)*K(muao,mupso)*Bi(m+.5,K(muai,mupsi)*a)*D(mupso)) / D(mupso)/rs(x0,y0,z0)**(1/2)/(
                                                      -1*D(mupsi)* a*Bi(m+1.5,K(muai,mupsi)*a)*K(muai,mupsi)*Bk(m+.5,K(muao,mupso)*a) 
                                                      -1*D(mupsi)*Bi(m+.5,K(muai,mupsi)*a)*m*Bk(m+.5,K(muao,mupso)*a) 
                                                      -1*D(mupso)*a*Bk(m+1.5,K(muao,mupso)*a)*K(muao,mupso)*Bi(m+.5,K(muai,mupsi)*a) 
                                                      +1*D(mupso)*Bk(m+.5,K(muao,mupso)*a)*m*Bi(m+.5,K(muai,mupsi)*a))

def foutBm (p,muao,muai,mupso,mupsi,a, x0, y0, z0, x, y, z):
    h = 0
    for m in range(0,p):
        h += (Bm(m,muao,muai,mupso,mupsi,a,x0, y0, z0)*Bk(m+.5,K(muao,mupso)*rs(x,y,z))/(rs(x,y,z))**(1/2))*P(0,m,costeta(x0,y0,z0,x,y,z))
    return h

def fout (p,muao,muai,mupso,mupsi,a, x0, y0, z0, x, y, z):
    return Green(muao,mupso, x0, y0, z0, x, y, z) + foutBm(p,muao,muai,mupso,mupsi,a, x0, y0, z0, x, y, z)

def foutTajada (p,muao,muai,mupso,mupsi,a, x0, y0, z0, x, y, z, z_inc):
    ze = 2.4/mupso    
    z00 = 1/mupso
    foutTemp = 0
    for m in range(-1,2):
        z0p = -z_inc + 2*m * s +4*m *ze + z00
        z0n = -z_inc -2*ze -z00 + 2*m *s + 4*m *ze
        foutTemp += fout(p,muao,muai,mupso,mupsi,a,x0,y0,z0p,x,y,z) - fout(p,muao,muai,mupso,mupsi,a,x0,y0,z0n,x,y,z)
    return foutTemp

def AAprox (n):
    if n > 1:
        A = 504.332889 - 2641.00214 * n + 5923.699064 * n**2 - 7376.355814 * \
            n**3 + 5507.53041 * n**4 - 2463.357945 * n**5 + \
            610.956547 * n**6 - 64.8047 * n**7
    if n <= 1:
        A = 3.084635 - 6.531194 * n + 8.357854 * \
            n**2 - 5.082751 * n**3 + 1.171382 * n**4
    return A


def Tcw (rho, mua, mups, s_thickness, n=1.4, mNum=10):
    
    z0 = 1/mups
    ze = 2*AAprox(n)*Dcomp(mua,mups)

    def z (m, typez):
        if typez == 1: return s_thickness*(1-2*m) - 4*m*ze - z0
        if typez == 2: return s_thickness*(1-2*m) - (4*m - 2)*ze - z0
        else: return np.nan   

    def sqrtTerm (m, typez):
        return np.sqrt( (mua * (np.power(rho,2) + np.power(z(m,typez),2)) ) / Dcomp(mua,mups) )

    def term1 (m, typez):
        return z(m,typez) * np.power(np.power(rho,2)+np.power(z(m,typez),2),-3/2)

    def term2 (m, typez):
        return 1 + sqrtTerm(m,typez)
    
    def term3 (m, typez):
        return np.exp( -sqrtTerm(m,typez))

    sumTerm = 0

    for m in range(-mNum, mNum+1):
        sumTerm += term1(m,1) * term2(m,1) * term3(m,1) - term1(m,2) * term2(m,2) * term3(m,2)

    return (1/(4*math.pi)*sumTerm)

def Tcw_muaSol (TcwIn, rho, mups, s_thickness, n=1.4, mNum=10, muaMax=1):
    
    def funcOpt (mua):
        return TcwIn - Tcw (rho, mua, mups, s_thickness, mNum)
    return optimize.root_scalar(funcOpt, bracket=[0, muaMax]).root


def meanPathLengthT (rho, mua, mups, s_thickness, n=1.4, mNum=10):
    z0 = 1/mups
    ze = 2*AAprox(n)*Dcomp(mua,mups)

    def z (m, typez):
        if typez == 1: return s_thickness*(1-2*m) - 4*m*ze - z0
        if typez == 2: return s_thickness*(1-2*m) - (4*m - 2)*ze - z0
        else: return np.nan
    
    def sqrtTerm (m, typez):
        return np.sqrt( (mua * (np.power(rho,2) + np.power(z(m,typez),2)) ) / Dcomp(mua,mups) )
        
    def term1 (m, typez):
        return z(m,typez) * np.power(np.power(rho,2)+np.power(z(m,typez),2),-1/2)
    
    def term2 (m, typez):
        return np.exp( -sqrtTerm(m,typez))
    
    def term3 (mau,mups):
        return 1/(8*math.pi * Dcomp(mua,mups) * Tcw (rho, mua, mups, s_thickness, n=n, mNum=mNum))        
    
    sumTerm = 0

    for m in range(-mNum, mNum+1):
        sumTerm += term1(m,1)*term2(m,1) - term1(m,2)*term2(m,2)
    
    return term3(mua,mups) * sumTerm
    
