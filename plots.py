# -*- coding: utf-8 -*-
"""
Created on Thu Nov 05 23:56:57 2015

@author: MPA Admin
"""

ek = load('ek.npy')
k = ek[:,0]
ek_min=ek[:,2]
ek_max=ek[:,3]
ek=ek[:,1]

k1 = arange(0.2,0.8,1e-1)
f = lambda k: 170*k**(-0.8)

loglog(k, ek, 'b')
#log(k1,f(k1), 'k', linewidth=0.7)
fill_between(k, ek_min, ek_max, facecolor='#C0C0C0',antialiased=True,alpha=0.5)
xlabel(r'$k \ [\mathrm{kpc}^{-1}]$')
ylabel(r'$E_B(k)$')
ylim(ymin=10,ymax=2e3)

text(0.015,40,r'$\mathrm{spectral\ index} \approx 3$')
text(0.015,25,r'$B_{centr} \approx 25 \ \mathrm{\mu G}$')
text(0.015,15,r'$\lambda_B \approx 3 \ \mathrm{kpc}$')
