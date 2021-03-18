# -*- coding: utf-8 -*-
"""
Created on Wed Nov 04 16:37:50 2015

@author: MPA Admin
"""

from numpy import *
import matplotlib.pyplot as plt
from faraday_lib import *
from matplotlib.colors import LogNorm
import sys
import ConfigParser
from optparse import OptionParser
import scipy.optimize as optimize
from scipy.special import jv, gamma
from scipy.ndimage.filters import gaussian_filter

config = ConfigParser.ConfigParser()
config.read('./params.cfg')

regime = config.get("input", "regime")
pds_stat_path = config.get("input", "pds_stat_path")
pds_out_path = config.get("output", "pds_out_path")
source = config.get("bmod_mf_general", "source")
if source == '3c449':
    kpc_px = config.getfloat("bmod_mf_3c449", "kpc_px")
elif source=='hydra':
    kpc_px = config.getfloat("bmod_mf_hydra", "kpc_px")

pdsk = load(pds_out_path)
pdsk_st = load(pds_stat_path)
kr, pds = pdsk[:,0], pdsk[:,1]
kr_st, pds_st, pds_st_min, pds_st_max = pdsk_st[:,0], pdsk_st[:,1], pdsk_st[:,2], pdsk_st[:,3]

#params0 = [0.1,3,5,0.2]
#sigma = 0.5*(pds_max-pds_min)
#pds_fit_params = optimize.curve_fit(PDS, kr, pds, params0, sigma)[0]
#print 'fit params: '+str(pds_fit_params)

pds_sm = gaussian_filter(kr**3*pds, 3,mode='mirror') / kr**3

corr_f = lambda p: (2**(p/2) *
            special.gamma(3-p/2)/special.gamma(3))
#lnkr, lnpds = log(kr), log(PDS(kr, *pds_fit_params))
lnkr, lnpds = log(kr), log(pds_sm)
ploc = -(lnpds[2:]-lnpds[:-2])/(lnkr[2:]-lnkr[:-2])
pds_corr = pds_sm[1:-1] / corr_f(ploc)

pds_min = (1 + (pds_st_min[1:-1]-pds_st[1:-1])/pds_st[1:-1]) * pds_corr
pds_max = (1 + (pds_st_max[1:-1]-pds_st[1:-1])/pds_st[1:-1]) * pds_corr

kmin = 0.001
kmax = kr[-1]
fk = (kmax/kmin)**(1./(100-1))
k = kmin * fk**arange(100)

if regime == '2d':
    amp = sqrt(2*pi*kr**2*pds)
#    amp_fit = sqrt(2*pi*kr**2*PDS(kr,*pds_fit_params))
    amp_min = sqrt(2*pi*kr**2*pds_min)
    amp_max = sqrt(2*pi*kr**2*pds_max)
elif regime == '3d':
    amp = sqrt(4*pi*kr**3*pds)
    amp_fit = sqrt(4*pi*kr**3*pds_sm)#PDS(kr,*pds_fit_params))
    amp_corr = sqrt(4*pi*kr[1:-1]**3*pds_corr)
#    AMP = sqrt(4*pi*kr**3*PDS(kr, C,p1,p2,kb))
    amp_min = sqrt(4*pi*kr[1:-1]**3*pds_min)
    amp_max = sqrt(4*pi*kr[1:-1]**3*pds_max)

if regime == '2d':
    ek = 2*pi*kr*pds
#    amp_fit = sqrt(2*pi*kr**2*PDS(kr,*pds_fit_params))
    ek_min = 2*pi*kr*pds_min
    ek_max = 2*pi*kr*pds_max
elif regime == '3d':
    ek = 4*pi*kr**2*pds
    ek_fit = 4*pi*kr**2*pds_sm#PDS(kr,*pds_fit_params))
    ek_corr = 4*pi*kr[1:-1]**2*pds_corr
#    AMP = sqrt(4*pi*kr**3*PDS(kr, C,p1,p2,kb))
    ek_min = 4*pi*kr[1:-1]**2*pds_min
    ek_max = 4*pi*kr[1:-1]**2*pds_max

dx = log(kr[1]/kr[0])
B0 = sqrt(4*pi*(kr[1:-1]**3*pds_corr*dx).sum())
print 'B0:', B0

lam = (kr[1:-1]**2*pds_corr*dx).sum()/(kr[1:-1]**3*pds_corr*dx).sum()
print 'lamB:', lam*kpc_px

fig = plt.figure()
ax1 = fig.add_subplot('111')

kr_ek = zeros((len(kr)-2,4))
kr_ek[:,0] = kr[1:-1]/kpc_px
kr_ek[:,1] = ek_corr*kpc_px
kr_ek[:,2] = ek_min*kpc_px
kr_ek[:,3] = ek_max*kpc_px
save('results/HydraA/ek.npy',kr_ek)

#ax1.loglog(kr/kpc_px, ek)
#ax1.loglog(kr/kpc_px, ek_fit, 'b--')
ax1.loglog(kr[1:-1]/kpc_px, ek_corr*kpc_px, 'b')
ax1.fill_between(kr[1:-1]/kpc_px, ek_min*kpc_px, ek_max*kpc_px,
                 facecolor='#C0C0C0',antialiased=True,alpha=0.5)
ax1.set_xlabel(r'$k \ [\mathrm{kpc}^{-1}]$')

#ax1.set_ylabel(r'$\mathrm{amplitude} \ [\mathrm{\mu G}]$')
ax1.set_ylabel(r'$E(k)$')
#ax1.set_ylim(ymin=1e-2)

plt.show()
