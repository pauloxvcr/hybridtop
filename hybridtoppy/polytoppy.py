import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib as mpl

from grandpy.GRANDpy import PlotBoundary

class HybridTop:
    def __init__(self, VolFrac : float, P, MatIntFncEl, MatIntFncBar, z1Ini = None,z2Ini = None,zMin=0.0,zMax=1.0,Tol = 0.01, MaxIter = 150, OCMove = 0.2, OCEta = 0.5, penal = 1.0):
        self.VolFrac = VolFrac
        self.P = P

        self.z1 = z1Ini
        self.z2 = z2Ini
        self.MatIntFncEl = MatIntFncEl
        self.MatIntFncBar = MatIntFncBar
        self.zMin=zMin
        self.zMax=zMax
        self.Tol = Tol
        self.MaxIter = MaxIter
        self.OCMove = OCMove
        self.OCEta = OCEta
        self.penal = penal

    def set_fem(self,fem, ratioelembar=0.5):
        self.fem = fem
        TotalVolume = sum(fem.ElemArea)  * self.VolFrac
        self.z1 = ratioelembar*self.VolFrac*np.ones(self.fem.NElem)
        self.z2 = (1-ratioelembar)*TotalVolume/fem.SumLenBars*np.ones(self.fem.NBars)

    def Opt(self, method = None):
        if method is None:
            method = [1,10,5]
        Iter = 0
        Tol = self.Tol * (self.zMax - self.zMin)
        Change = 2 * Tol
        z1 = self.z1
        z2 = self.z2
        P = self.P
        try: self.fem
        except: raise RuntimeError("you don't defined the finite element yet")
        E1,dE1dy1,V1,dV1dy1 = self.MatIntFncEl((P @ z1).flatten(),self.penal) #verificar a multiplicação
        E2,dE2dy2,V2,dV2dy2 = self.MatIntFncBar (z2)
        # FigHandle, FigData = InitialPlot(fem,V)
        while(Iter<self.MaxIter) and (Change>Tol):
            E = np.append(E1,E2)
            if Iter <method[1] or Iter % method[2] == 0:
                self.fem.NonLinearFixedPoint(E)
            else:
                self.fem.FEAnalysis(E)
            Iter = Iter+1
            #Compute cost functionals and analysis sensitivities
            f,dfdE1,dfdV1,dfdE2,dfdV2 = self.ObjectiveFnc(E1,E2,V1,V2)
            g,dgdE1,dgdV1,dgdV2,dgdE2 = self.ConstraintFnc(E1,E2,V1,V2)
            #Compute design sensitivities for Bidimensional
            dfdz1 = P.transpose() @ (dE1dy1*dfdE1 + dV1dy1*dfdV1)
            dgdz1 = P.transpose() @ (dE1dy1*dgdE1 + dV1dy1*dgdV1)
            #Compute design sensitivities for Bars
            dfdz2 = (dE2dy2*dfdE2 + dV2dy2*dfdV2)
            dgdz2 = (dE2dy2*dgdE2 + dV2dy2*dgdV2)
            #Update design variable and analysis parameters

            z1,z2,Change = self.UpdateScheme(g,dfdz1,dgdz1,z1,dfdz2,dgdz2,z2)
            E1,dE1dy1,V1,dV1dy1= self.MatIntFncEl((P @ z1).flatten(), self.penal)
            E2, dE2dy2, V2, dV2dy2 = self.MatIntFncBar(z2)
            print(f'It: {Iter} \t Objective: {f} \tChange: {Change}')

        self.z1 = z1
        self.z2 = z2
        self.V1 = V1
        self.V2 = V2

    def ObjectiveFnc(self,E1,E2,V1,V2):
        #E = np.append(E1,E2)
        fem = self.fem
        U = fem.U
        f = np.dot(fem.F,U)
        #for bidimensional
        temp = np.cumsum(-U[fem.i1]* fem.k1* U[fem.j1])
        temp = temp[np.cumsum(fem.ElemNDof**2)-1]
        dfdE1 = np.append(temp[0],temp[1:len(temp)]-temp[0:-1])
        dfdV1 = np.zeros(np.size(V1))
        #for unidimensional
        temp = np.cumsum(-U[fem.i2] * fem.k2 * U[fem.j2])
        temp = temp[np.cumsum(np.ones(fem.NBars,dtype=int)*16) - 1]
        dfdE2 = np.append(temp[0], temp[1:len(temp)] - temp[0:-1])
        dfdV2 = np.zeros(np.size(V2))

        return f,dfdE1,dfdV1,dfdE2,dfdV2

    def ConstraintFnc(self,E1,E2,V1,V2):
        fem = self.fem
        g = (sum(fem.ElemArea*V1)+sum(fem.LenBars * V2))/sum(fem.ElemArea) - self.VolFrac
        #Bidimensional
        dgdE1 = np.zeros(np.size(E1))
        dgdV1 = fem.ElemArea/sum(fem.ElemArea)
        #Barras
        dgdV2 = fem.LenBars / sum(fem.ElemArea)
        dgdE2 = np.zeros(np.size(E2))

        return g,dgdE1,dgdV1,dgdV2,dgdE2

    def UpdateScheme(self,g,dfdz1,dgdz1,z1,dfdz2,dgdz2,z2): # ver substitui por uma função scypy
        zMin = self.zMin
        zMax = self.zMax
        move = self.OCMove*(zMax-zMin)
        eta = self.OCEta

        dfdz = np.append(dfdz1,dfdz2)
        dgdz = np.append(dgdz1,dgdz2)
        z0 = np.append(z1,z2)

        l1 = 0
        l2 = 1e6
        while l2-l1 > 1e-4:
            lmid = 0.5*(l1+l2)
            B = -(dfdz/dgdz)/lmid
            zCnd = zMin+(z0-zMin)*(B**eta)
            zNew = np.maximum(np.maximum(np.minimum(np.minimum(zCnd,z0+move),zMax),z0-move),zMin)
            if (g + (dgdz @ (zNew-z0)) > 0): l1 = lmid
            else: l2 = lmid
        Change = max(abs(zNew-z0))/(zMax-zMin)
        z1 = zNew[0:self.fem.NElem]
        z2 = zNew[self.fem.NElem:len(zNew)]

        return z1,z2,Change

    def polyPlot(self, tol=0.001):
        fig, ax = plt.subplots(nrows=1, ncols=2,gridspec_kw={'width_ratios': [10, 1]})
        BiDimensionalElements = self.fem.BiDimensionalElements
        PlotBoundary(self.fem.Mesher.Node, self.fem.Mesher.Element, ax[0])

        num_polygons = self.fem.NElem
        c = 1 - self.V1
        c[c<0] = 0
        for i in range(num_polygons):

            polygon = Polygon(self.fem.Node[BiDimensionalElements[i].eNode],color=[c[i],c[i],c[i]])
            ax[0].add_patch(polygon)
        ax[0].autoscale()

        Bars = self.fem.Bars
        c = self.V2/max(self.V2)
        for i in range(self.fem.NBars):
            if self.V2[i]>tol:
                line = Line2D([self.fem.Node[Bars[i].bNode[0], 0], self.fem.Node[Bars[i].bNode[1], 0]], [self.fem.Node[Bars[i].bNode[0], 1], self.fem.Node[Bars[i].bNode[1], 1]],
                              color=[c[i],0,1-c[i]],
                              linewidth=4*np.sqrt(c[i]))
                ax[0].add_line(line)

        ax[0].set_aspect('equal')
        ax[0].axis('off')
        # Color bar
        x = np.linspace(0, 1, 256)
        newcolormap = np.column_stack((x, np.zeros(len(x)), 1 - x, np.ones(len(x))))
        newcolormap = ListedColormap(newcolormap)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=newcolormap), cax=ax[1])



    def save_result(self):
        return ResultPolyTruss(self.fem.Mesher, self.V1, self.V2,self.z1,self.z2)

# -------------------------------------- Result ----------------------------------------------------------------


class ResultPolyTruss:
    def __init__(self,Mesher,V1,V2,z1,z2):
        self.Mesher = Mesher
        self.V1 = V1
        self.V2 = V2
        self.z1 = z1
        self.z2 = z2

def plotResult(Result,tol=0.001):
    fig, ax = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [10, 1]})
    BiDimensionalElements = Result.Mesher.Element
    num_polygons = len(Result.Mesher.Element)
    c = 1 - Result.V1
    c[c < 0] = 0
    for i in range(num_polygons):
        polygon = Polygon(Result.Mesher.Node[BiDimensionalElements[i].eNode], color=[c[i], c[i], c[i]])
        ax[0].add_patch(polygon)
    ax[0].autoscale()

    Bars = Result.Mesher.Bars
    c = Result.V2 / max(Result.V2)
    for i in range(len(Result.Mesher.Bars)):
        if Result.V2[i] > tol:
            line = Line2D([Result.Mesher.Node[Bars[i].bNode[0], 0], Result.Mesher.Node[Bars[i].bNode[1], 0]],
                          [Result.Mesher.Node[Bars[i].bNode[0], 1], Result.Mesher.Node[Bars[i].bNode[1], 1]],
                          color=[c[i], 0, 1 - c[i]],
                          linewidth=np.log(1000 ** c[i]))
            ax[0].add_line(line)

    ax[0].set_aspect('equal')
    ax[0].axis('off')
    # Color bar
    x = np.linspace(0, 1, 256)
    newcolormap = np.column_stack((x, np.zeros(len(x)), 1 - x, np.ones(len(x))))
    newcolormap = ListedColormap(newcolormap)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=newcolormap), cax=ax[1])
























