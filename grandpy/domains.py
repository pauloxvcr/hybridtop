import numpy as np
from polymesherpy.polymesherpy import PolyMshr_PlotMsh
import matplotlib.pyplot as plt

def StructDomain(Nx,Ny,Lx,Ly):
    x,y = np.meshgrid(np.linspace(0,Lx,Nx+1),np.linspace(0,Ly,Ny+1))
    Node = np.column_stack((x.flatten(),y.flatten()))
    Elem = []
    for i in range(0,Ny):
        for j in range(0,Nx):
            n1 = i*(Nx+1)+j
            n2 = (i+1)*(Nx+1) + j
            Elem.append([n1,n1+1,n2+1,n2])
    return Node, Elem,[],[]

'''Node, Element, supp,load = StructDomain(60,20,3,1)
PolyMshr_PlotMsh(Node,Element)
plt.show()'''