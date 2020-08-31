###
'''
This algorithm is a translation of a MatLab algorithm that can by found with the algorithm article on:
https://paulino.ce.gatech.edu/software.html

Reference of the algorithm article:
Zegard T, Paulino GH (2014) GRAND — Ground structure based topology optimization for arbitrary 2D domains using
MATLAB. Structural and Multidisciplinary Optimization 50:861–882. https://doi.org/10.1007/s00158-014-1085-z
'''
###
import numpy as np
from .generateGsC import generateGS
from scipy.sparse import csr_matrix, hstack
from scipy.optimize import linprog
from scipy.sparse import find

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
#=== MESH GENERATION LOADS/BCS ==========================================

def Grand(Mesher,Bars,kappa=1.0,ColTol = 0.999999,CutOff = 0.001,Ng = 50):
    Node,Elem,Supp,Load = Mesher.get()
    Nn = np.size(Node,0)
    Ne = len(Elem)
    Nb = np.size(Bars,0)
    #------GetSupports
    NSupp = np.size(Supp, 0)
    FixedDofs = np.append((2 * Supp[0:NSupp, 0])[Supp[0:NSupp, 1].astype(bool)],
                          (2 * Supp[0:NSupp, 0] + 1)[Supp[0:NSupp, 2].astype(bool)])
    AllDofs = np.arange(2*Nn)
    FreeDofs = np.setdiff1d(AllDofs, FixedDofs)
    # -----------GetVectorF
    NLoad = np.size(Load, 0)
    F = np.zeros(2 * Nn)
    F[2 * Load[0:NLoad, 0].astype(int)] = Load[0:NLoad, 1]
    F[(2 * Load[0: NLoad, 0].astype(int)) + 1] = Load[0: NLoad, 2]
    F = F[FreeDofs]
    #-------------- GetMatrixBt
    D = np.column_stack( ( Node[Bars[:,1],0]-Node[Bars[:,0],0] , Node[Bars[:,1],1]-Node[Bars[:,0],1] ) )
    L = np.sqrt(D[:,0]**2+D[:,1]**2)
    D[:,0] = D[:,0]/L
    D[:,1] = D[:,1]/L
    i = np.column_stack((2*Bars[:,0],2*Bars[:,0]+1,2*Bars[:,1],2*Bars[:,1]+1)).flatten()
    j = np.repeat(np.arange(Nb).reshape(Nb,1),4,1).flatten()
    k = np.column_stack((-D,D)).flatten()
    BT = csr_matrix((k, (i, j)), shape=(2*Nn, Nb))
    BT = BT[FreeDofs,:]

    #-----------------------------
    BTBT = hstack((BT,-BT))
    LL = np.append(L,kappa*L)
    #-----------
    #del BT
    #del L
    #----------
    Result = linprog(c=LL, A_ub=None,b_ub=None,A_eq=BTBT,b_eq=F,bounds=(0,None))
    print(f'Solution {Result.success}')
    S = Result.x
    vol = Result.fun
    S= S.reshape(int((len(S)/2)),2,order='F')
    A = S[:,0]+kappa*S[:,1]
    N = S[:,0]-S[:,1]
    Teste = BT @ N
    PlotGroundStructure(Node,Elem,Bars,A,CutOff)
    return A

def PlotGroundStructure(Node,Elem,Bars,A,CutOff=0.001):
    fig, ax = plt.subplots()
    PlotBoundary(Node, Elem, ax)
    A = A/max(A)
    for i in range(np.size(Bars,0)):
        if A[i] > CutOff:
            line = Line2D([Node[Bars[i,0], 0], Node[Bars[i,1], 0]],
                          [Node[Bars[i,0], 1], Node[Bars[i,1], 1]],
                          color=[A[i], 0, 1 - A[i]],
                          linewidth= 4 * np.sqrt( A[i]))
            ax.add_line(line)

    ax.autoscale()
    ax.set_aspect('equal')
    ax.axis('off')

from scipy.sparse import lil_matrix, find

def PlotBoundary(Node,Elem,ax):
    # Get number of nodes, elements and edges (nodes) per element
    Nn= np.size(Node,0)
    Ne = len(Elem)
    NpE = [len(element) for element in Elem]
    Face = lil_matrix((Nn,Nn))
    for i in range(Ne):
        MyFace = np.stack((Elem[i],np.append(Elem[i][1:len(Elem)],Elem[i][0])))
        for j in range(NpE[i]):
            if Face[MyFace[0,j],MyFace[1,j]] == 0: #New  edge
                Face[MyFace[0, j], MyFace[1, j]] = 1
                Face[MyFace[1, j], MyFace[0, j]] = -1
            elif np.isnan(Face[MyFace[0, j], MyFace[1, j]]):
                raise('error')
            else:
                Face[MyFace[0, j], MyFace[1, j]] = np.nan
                Face[MyFace[1, j], MyFace[0, j]] = np.nan
    B1,B2,x = find(Face>0)
    x = np.stack((Node[B1,0],Node[B2,0]))
    y = np.stack((Node[B1,1],Node[B2,1]))
    ax.plot(x,y,color='k')
