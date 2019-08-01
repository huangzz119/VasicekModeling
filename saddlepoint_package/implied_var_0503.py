import numpy as np
import pandas as pan
from scipy.stats import norm
from scipy.optimize import fsolve


n_obligor = 100
unpd = 0.1
rho = 0.2
ead = np.arange(1, 101)
lgd = np.ones(n_obligor)
w = ead * lgd

alpha = 0.95


GL = pan.read_csv('GL-quad.csv')
Y = GL.abscissas.values.tolist()
GLweight = GL.weights.values.tolist()
Y_ = 5 * np.asarray(Y)

cpd = norm.cdf(( norm.ppf(unpd) - np.sqrt(rho) * Y_ ) / np.sqrt(1 - rho))
weight = norm.pdf(Y_)  * GLweight


def alpha_to_var(x):

    ctp = np.zeros(len(cpd))

    for i in np.arange(len(cpd)):
        pd = cpd[i]

        def cgf_fir(t):
            temp_first_num = w * pd * np.exp(w * t)
            temp_first_den = 1 - pd + pd * np.exp(w * t)
            return sum((temp_first_num / temp_first_den)) - x

        t = fsolve(cgf_fir, np.array([0.1]))

        # second derivative of conditional CGF
        sed_cgf_num = (1 - pd) * np.square(w) * pd * np.exp(w * t)
        sed_cgf_den = np.square(1 - pd + pd * np.exp(w * t))
        sed_cgf = sum((sed_cgf_num / sed_cgf_den))

        # conditional CGF
        cgf = sum(np.log(1 - pd + pd * np.exp(w * t)))

        if sed_cgf>0 and 2 * (x * t[0] - cgf)>0:

            z_w = t * np.sqrt(sed_cgf)
            z_l = np.sign(t) * np.sqrt(2 * (x * t[0] - cgf))
            tp = 1 - norm.cdf(z_l) + norm.pdf(z_l) * ((1 / z_w) - (1 / z_l))
            ctp[i] = tp
        else:
            ctp[i] = ctp[i-1]

    untp = 5 * sum( ctp * weight)

    return untp - (1-alpha)

var = fsolve(alpha_to_var, np.array([1200]))
print(var)


ces522a = np.zeros(len(cpd))
ces522c = np.zeros(len(cpd))
ctp = np.zeros(len(cpd))

for i in np.arange(len(cpd)):
    pd = cpd[i]
    muL = pd * sum(w)

    def cgf_fir(t):
        temp_first_num = w * pd * np.exp(w * t)
        temp_first_den = 1 - pd + pd * np.exp(w * t)
        return sum((temp_first_num / temp_first_den)) - var[0]
    t = fsolve(cgf_fir, np.array([0.1]))

    # second derivative of conditional CGF
    sed_cgf_num = (1 - pd) * np.square(w) * pd * np.exp(w * t)
    sed_cgf_den = np.square(1 - pd + pd * np.exp(w * t))
    sed_cgf = sum((sed_cgf_num / sed_cgf_den))

    # conditional CGF
    cgf = sum(np.log(1 - pd + pd * np.exp(w * t[0])))

    z_w = t * np.sqrt(sed_cgf)
    z_l = np.sign(t) * np.sqrt(2 * (var[0] * t[0] - cgf))

    es522a = muL*(1 - norm.cdf(z_l)) + norm.pdf(z_l) * ((var[0] / z_w) - (muL / z_l))
    es522c = muL*(1 - norm.cdf(z_l)) + norm.pdf(z_l) * ((var[0] / z_w) - (muL / z_l) + (muL-var[0])/(z_l**3) + 1 / (t[0]*z_w))
    ces522a[i] = es522a
    ces522c[i] = es522c

    tp = 1 - norm.cdf(z_l) + norm.pdf(z_l) * ((1 / z_w) - (1 / z_l))
    ctp[i] = tp

print(5 * sum(ces522a * weight) * 20)
print(5 * sum(ces522c * weight) * 20)
5 * sum( ctp * weight)

# 10   20   100

# method of ES533
def var_to_es533(var):
    ces533 = np.zeros(len(cpd))

    for i in np.arange(len(cpd)):
        pd = cpd[i]
        muL = pd*sum(w)

        def cgf_fir_(t):
            fir_cgf_num = w * pd * np.exp(w * t)
            fir_cgf_den = 1 - pd + pd * np.exp(w * t)
            fir_cgf = sum((fir_cgf_num / fir_cgf_den))

            sed_cgf_num = (1 - pd) * np.square(w) * pd * np.exp(w * t)
            sed_cgf_den = np.square(1 - pd + pd * np.exp(w * t))
            sed_cgf = sum((sed_cgf_num / sed_cgf_den))

            fir_cgf_ = fir_cgf + sed_cgf/fir_cgf
            return fir_cgf_ - var[0]

        t = fsolve(cgf_fir_, np.array([0.1]))

        cgf_den = 1 - pd + pd * np.exp(w * t)

        tir_num1 = (1 - pd) *  (w**3) * pd * np.exp(w * t)
        tir_num2 = 2* (1 - pd) * (w**3) * (pd**2) * np.exp(2*w * t)
        tir_cgf = sum(tir_num1/(cgf_den**2) - tir_num2/(cgf_den**3))

        sed_num = (1 - pd) * np.square(w) * pd * np.exp(w * t)
        sed_cgf = sum((sed_num / cgf_den**2))

        fir_num = w * pd * np.exp(w * t)
        fir_cgf = sum((fir_num / cgf_den))
        fir_cgf0 = sum((w * pd * np.exp(w * 0)) / (1 - pd + pd * np.exp(w * 0)))

        cgf = sum(np.log(1 - pd + pd * np.exp(w * t)))

        cgf_ = cgf + np.log(fir_cgf) - np.log(fir_cgf0)
        sed_cgf_ = sed_cgf + (tir_cgf*fir_cgf - sed_cgf**2)/(fir_cgf**2)

        z_w = t * np.sqrt(sed_cgf_)
        z_l = np.sign(t) * np.sqrt(2 * (var[0] * t[0] - cgf_))
        tp = 1 - norm.cdf(z_l) + norm.pdf(z_l) * ((1 / z_w) - (1 / z_l))
        es = muL/(1-alpha) * tp
        ces533[i] = es

    es533 = 5 * sum( ces533 * weight)

    return es533
es533 = var_to_es533(var)

print(es533)
























