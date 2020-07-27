import numpy as np
def simp(y):
    E = y
    V = y
    dEdy = np.ones(np.size(y))
    dVdy = np.ones(np.size(y))
    return E,dEdy,V,dVdy