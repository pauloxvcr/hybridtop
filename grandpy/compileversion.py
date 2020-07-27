import numpy as np
from numba import jit

@jit(nopython=True)
def selectRemoveFlag(RemoveFlag,I,Bars,Nn,Nb, ColTol,D,newD):
    p = 0
    m = 0
    for j in range(0, Nn):
        # Find I(p:q) - NEW bars starting @ node 'j' ( I está em ordem(por isso que fazia diferença la em cima) então p e q servem para achar as barras que saem do no j=p sendo q-1 o ultimo indice
        for p in range(p, len(I)):
            if I[p] >= j:
                break
        for q in range(p, len(I)):
            if I[q] > j:
                break
        if I[q] > j:  # faz diferença no ultimo nó analisado
            q = q - 1

        if I[p] == j:  # dupla garantia?
            # Find BARS(m:n) - OLD bars starting @ node 'j'
            for m in range(0, Nb):
                if Bars[m, 0] >= j:  # barras também estão em ordens
                    break
            for n in range(m, Nb):
                if Bars[n, 0] > j:
                    break
            if Bars[n, 0] > j:
                n = n - 1
            if Bars[n, 0] == j:
                # Dot products of old vs. new bars. If ~collinear: mark
                C = maxColumn(D[m:n + 1, :] @ newD[p:q + 1, :].T)  # possível erro
                RemoveFlag[p + ccoltol(C,ColTol)] = 1
    return RemoveFlag
@jit(nopython = True)
def maxColumn(array):
    result = np.zeros(array.shape[1])
    for i in range(array.shape[1]):
        result[i] = np.max(array[:,i])

    return result
@jit(nopython = True)
def ccoltol(C,Coltol):
    result = []
    for i in range(C.size):
        if C[i]>Coltol:
            result.append(i)
    return np.array(result)