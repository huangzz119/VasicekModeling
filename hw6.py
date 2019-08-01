

import numpy as np
from scipy.stats import norm

k = np.linspace(0.01,0.08,8)

a = 0.25
b = 0.5
v = 0.25
rho = -0.25

f0 = 0.04
T= 1

q = v/(a*(1-b)) * (f0**(1-b) - k**(1-b))
Dq = np.log( (np.sqrt(1-2*rho*q+q**2)+q-rho) / (1-rho) )

fm = np.sqrt(f0*k)
y1 = b/fm
y2 = -(b*(1-b))/(fm**2)
cf = fm**b
e = T*(v**2)

t1 = (2*y2 - y1**2 + 1/(fm**2)) / 24 * (a*cf/v)**2
t2 = ((rho*y1)/4) * ((a*cf)/v)
t3 = (2-3*rho**2)/24

sigma = v * (np.log(f0/k) / Dq ) * (1 + (t1+t2+t3)*e )

d1 = ( np.log(f0/k) + 0.5*(sigma**2)*T ) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)

c = ( f0 * norm.cdf(d1) - k * norm.cdf(d2))

