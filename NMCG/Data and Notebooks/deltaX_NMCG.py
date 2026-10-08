"""
Created on Thu Jan 28 11:08:30 2021

@author: Andrés
"""

#######################
#### M O D U L O S ####
#######################

import numpy as np
#import math 
import matplotlib.pyplot as plt
import plt_conf as conf
from scipy.optimize import least_squares

conf.general()
#######################

#######################
## F U N C I O N E S ##
#######################

# Parámetros; poner los valores idoneos

q0 = -0.53
j0 = 1.0
h0 = 0.71

a = 1.0
h = 1
k1 = 0.1
k2 = 0.3
f = 1.0
k = 1.0


zz = np.linspace(0.01,1.0,1000)

def H(z):
    a = (1.0+(1.0+q0)*z+0.5*(-1.0*pow(q0,2.0)+j0)*pow(z, 2.0))
    return a

def dH(z):
    v = -1.0*((1.0+q0+(j0-pow(q0,2.0))*z)*(1.0+(1.0+q0)*z+0.5*(j0-pow(q0,2.0))*pow(z,2.0)))
    return v
    
def ddH(z):
    c = -1.0*((1+(1.0+q0)*z+0.5*(j0-pow(q0,2.0))*pow(z,2.0))*(-1.0*pow(1.0+q0+j0*z-pow(q0,2.0)*z,2.0)-(j0-pow(q0,2.0)*(1.0+(1.0+q0)*z+0.5*(j0-pow(q0,2.0))*pow(z,2.0)))))
    return c

def r(z):
    c = 0.33*pow(1.0+z,3.0)+0.66
    return c

def dr(z):
    c = -0.99*pow(1.0+z,2.0)*(1.0+(1.0+q0)*z+0.5*(j0-pow(q0,2.0))*pow(z,2.0))
    return c


#def d(z):
#    a = 1.0 / (1.0 + z)
#    num = f * a**9 * (
#        3 * a * (2 * H(z)**3 - ddH(z)) * r(z)
#        + (a * H(z)**2 + a * dH(z)) * dr(z))
#    den = 108 *H(z)* k * (a * H(z)**2 + a * dH(z))**4
#   return den

def d(z):
    w = (1.0*f*pow(1/(1+z),9.0)*(3.0*1.0/(1.0+z)*(2.0*pow(H(z),3.0)-ddH(z))*r(z)+(1.0/(1.0+z)*pow(H(z),2.0)+1.0/(1.0+z)*dH(z))*dr(z)))/(108.0*k*pow((1.0/(1.0+z)*pow(H(z),2.0)+1.0/(1.0+z)*dH(z)),4.0))
    return w

vect1=[]

for i in range(len(zz)):
    aa = d(zz[i])
    vect1.append(aa)

def d1(p, z):
    x = (1.0-p[0])*((p[1]*pow(1.0+z,p[1])*(1.0+pow(1.0+z,2.0*p[1]))/(-1.0+pow(1.0+z,2.0*p[1]))**2.0)-2.0*p[2]*pow(1.0+z,2.0*p[2])/((-1.0+pow(1.0+z,2.0*p[2]))**2.0)+p[1]/(1.0+p[0]*(-1.0+pow(1.0+z,p[1]))))
    return x

## p[0] es xi
## p[1] es n
## p[2] es r
#############################
##### MINIMOS CUADRADOS #####
#############################

param_list = []

def residuos(p, z, vect1):
    y_modelo = d1(p, zz)
    plt.clf()
    plt.plot(zz,vect1,'o',zz,y_modelo,'r-')
    plt.pause(0.05)
    param_list.append(p)
    return y_modelo - vect1

parametros_iniciales=[1.0, 1.0, 1.0]  # Ajusta
res = least_squares(residuos, parametros_iniciales, args=(zz, vect1),  verbose=1)

# Estos son los parámetros hallados:
print('parámetros hallados')
print(res.x)


#Calculamos la matriz de covarianza "pcov"
def calcular_cov(res,vect1):
    U, S, V = np.linalg.svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * S[0]
    S = S[S > threshold]
    V = V[:S.size]
    pcov = np.dot(V.T / S**2, V)#

    s_sq = 2 * res.cost / (len(vect1) - len(res.x))
    pcov = pcov * s_sq
    return pcov

pcov = calcular_cov(res,vect1)

# De la matriz de covarinza podemos obtener los valores de desviación estándar
# de los parametros hallados
pstd = np.sqrt(np.diag(pcov))

print('Parámetros hallados (con incertezas):')
for i,param in enumerate(res.x):
    print('parametro[{:d}]: {:5.3f} ± {:5.3f}'.format(i,param,pstd[i]/2))

y_modelo = d1(res.x, zz)

np.savetxt('deltaX.txt', y_modelo)

#######################
### G R A F I C A S ###
#######################    

figDeltaT1 = plt.figure()
plt.plot(zz, vect1,  'b o', markersize=4, label='$\delta_{NMCG}$')
plt.plot(zz, y_modelo, 'r-', label='$\delta_{X}$')
plt.xscale('log',base=10)
plt.xlabel("z", fontsize=13)
plt.ylabel("$\delta(z)$", fontsize=13)
plt.legend(loc='lower left', fontsize=13)
#plt.tight_layout()

#figDeltaT1.savefig('deltaNMCG_X_q0var.pdf')
plt.show()

#vect3=[] # Este vector es la función d1 evaluada con los resultados del diff evol #

#for i in range(len(zz)):
#    aa = d1(zz[i], result.x[0], result.x[1])
#    vect3.append(aa)

#figCompHz1s = plt.figure()
#plt.plot(zz, vect1, 'b-', linewidth = 1, label = '$\delta(z)$')
#plt.plot(zz, vect3, 'r-', linewidth = 1, label = '$\delta_1(z, \Xi)$ 2.37')
#plt.legend(loc = 1, numpoints = 1, fontsize = 9)
#plt.xlabel("z", fontsize = 11)
#plt.ylabel("$\delta(z)$", fontsize = 11)
#figCompHz1s.savefig('deltazeta.pdf')
#plt.show()


#################################
### T E S T  C H I ^ 2       ###
#################################

from scipy.stats import chi2

obs = np.array(vect1)
chi2_val = np.sum((obs - y_modelo)**2 / np.abs(y_modelo))   # chi^2 de Pearson
gl = len(obs) - len(res.x)                                    # grados de libertad
print('chi^2 = {:.5e}, chi^2 reducido = {:.5e}, p-valor = {:.5f}'.format(chi2_val, chi2_val/gl, chi2.sf(chi2_val, gl)))

