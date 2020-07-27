import numpy as np
eps = 1e-4
def simp(y,param):
    penal = param
    E = eps + (1 - eps) * (y ** penal)
    V = y
    dEdy = (1 - eps) * penal * (y ** (penal - 1))
    dVdy = np.ones(np.size(y))
    return E,dEdy,V,dVdy
def simp_h(y,param):
    penal = param[0]
    beta = param[1]
    h = 1 - np.exp(-beta * y) + y * np.exp(-beta)
    E = eps + (1 - eps) * h ** penal
    V = h
    dhdy = beta * np.exp(-beta * y) + np.exp(-beta)
    dEdy = (1 - eps) * penal * h ** (penal - 1) ** dhdy
    dVdy = dhdy
    return E,dEdy,V,dVdy
def ramp(y,param):
    q = param
    E = eps + (1 - eps) *(y ** (1 + q * (1 - y)))
    V = y
    dEdy = ((1 - eps) * (q + 1)) / (q - q * y + 1) ** 2
    dVdy = np.ones((np.size(y)))
    return E, dEdy, V, dVdy
def ramp_h(y,param):
    q = param[0]
    beta = param[1]
    h = 1 - np.exp(-beta * y) + y * np.exp(-beta)
    E = eps + (1 - eps) * h / (1 + q * (1 - h))
    V = h
    dhdy = beta * np.exp(-beta * y) + np.exp(-beta)
    dEdy = ((1 - eps) * (q + 1)) / ((q - q * h + 1)** 2) * dhdy
    dVdy = dhdy
    return E, dEdy, V, dVdy