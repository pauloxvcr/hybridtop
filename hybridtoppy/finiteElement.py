import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize,root


from .auxiliarNumba import *

# -------------------------

class Generic():
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
        self.W = None
        self.N = None
        self.dNdxi = None
# --------------------------------- Elements
class BiDimensionalElement:
    def __init__(self,eNode):
        self.eNode = eNode
        self.size = eNode.size
        self.NDof = self.size*2
        self.eDof = np.append([2 * self.eNode], [2 * self.eNode + 1], axis=0).reshape((self.NDof, 1),order = 'F')

        self.Ue = np.zeros(2*self.size)
        self.sigma = np.zeros(3)
        self.sigmap = np.zeros(3)
        self.theta = 0

        self.Ke = np.zeros((2 * self.size, 2 * self.size))

    def LocalK(self,Node, ShapeFnc,Nu0,Ec,Et):
        nn = np.size(self.eNode)
        self.Ke,self.DB = LocalKbi(self.eNode,Node, ShapeFnc[nn].W, ShapeFnc[nn].dNdxi,Nu0, Ec,Et,self.sigmap,self.theta) #wraper comp

    def calcElemArea(self,Node):
        vx = Node[self.eNode, 0]
        vy = Node[self.eNode, 1]
        self.ElemArea = 0.5 * sum(vx * np.append(vy[1:len(vy)], vy[0]) - vy * np.append(vx[1:len(vx)], vx[0]))

    def calcsigma(self):
        self.sigma = calcsigma(self.DB,self.Ue,self.ElemArea).flatten() #wraper compilated

    def calctheta(self):
        self.sigmap, self.theta = calctheta(self.sigma) # wraper compilated
    def setUe(self,U):
        self.Ue = U[self.eDof] # fazer wrapper nao

# --------------------------------------
class Bar:
    def __init__(self,bNodes):
        self.bNode = bNodes
        self.BarType = 0
        self.eDof = np.append([2 * self.bNode], [2 * self.bNode + 1], axis=0).reshape((4, 1),order = 'F')


    def LocalKBars(self, Node,EA):
        x1 = Node[self.bNode[0],0]
        y1 = Node[self.bNode[0],1]
        x2 = Node[self.bNode[1], 0]
        y2 = Node[self.bNode[1], 1]

        l = ((x2-x1)**2+(y2-y1)**2)**0.5
        c = (x2 - x1) / l
        s = (y2 - y1) / l
        T = np.array([[c,s,0.0,0.0],[0.0,0.0,c,s]])
        Ke = EA/l * T.T  @ np.array([[1.0,-1.0],[-1.0,1.0]]) @ T
        self.Ke = Ke
        self.l = l
        self.T = T
    def calcdeformation(self):
        self.BarType = calcdeformation(self.T, self.l, self.Ue) #wrapper

    def setUe(self,U):
        self.Ue = U[self.eDof]
#--------------------------------------


class FiniteElementStructure:
    def __init__(self, Mesher, Nu0 = 0.2, Ec=1,Et = 0.1,EA = 10,Reg = 0):
        self.Mesher = Mesher

        self.Node = self.Mesher.Node
        self.NNode = np.size(self.Node, 0)
        self.Supp = self.Mesher.Supp
        self.Load = self.Mesher.Load



        self.Nu0 = Nu0
        self.Ec = Ec
        self.Et = Et
        self.EA = EA
        self.Reg = Reg

        #VERIFICAR SE ELEMENT VEM COMO ARRAY
        self.NElem = np.size(self.Mesher.Element, 0)
        #defineBiDimensionalElements
        self.BiDimensionalElements = []
        for i in range(self.NElem):
            self.BiDimensionalElements.append(BiDimensionalElement(self.Mesher.Element[i]))

        self.NBars = len(self.Mesher.Bars)
        #defineBars
        self.Bars = []
        for i in range(self.NBars):
            self.Bars.append(Bar(self.Mesher.Bars[i]))
            self.Bars[i].LocalKBars(self.Node,EA)



        self.TabShapeFnc()
        self.setElemArea() #!!
        self.setSystem()

        self.U = np.zeros(2 * self.NNode)
        self.update_k()
        self.LenBars = np.array([bar.l for bar in self.Bars])
        self.SumLenBars = sum(self.LenBars)

    def setElemArea(self):
        self.ElemArea = np.zeros(self.NElem)
        for el in range(0, self.NElem):
            self.BiDimensionalElements[el].calcElemArea(self.Node)
            self.ElemArea[el] = self.BiDimensionalElements[el].ElemArea

    def setSystem(self):
        #-------Bidimensional Elements ------------
        self.ElemNDof = 2 * np.array([element.size for element in self.BiDimensionalElements])
        self.i1 = np.zeros(sum(self.ElemNDof ** 2), dtype=int)
        self.j1 = self.i1.copy()
        self.e1 = self.i1.copy()
        self.k1 = np.zeros(sum(self.ElemNDof ** 2))
        index = 0
        for el in range(0, self.NElem):
            NDof = self.ElemNDof[el]
            eDof = np.append([2 * self.BiDimensionalElements[el].eNode], [2 * self.BiDimensionalElements[el].eNode + 1], axis=0).reshape((NDof, 1),order = 'F')
            I = np.repeat(eDof, NDof, axis=1)
            J = I.transpose()
            self.i1[index:index + NDof ** 2] = I.flatten(order = 'F')
            self.j1[index:index + NDof ** 2] = J.flatten(order = 'F')
            self.e1[index:index + NDof ** 2] = el
            index = index + NDof ** 2  # python open interval
        #-------Bars-------------
        self.i2 = np.zeros(self.NBars * (4 ** 2), dtype=int)
        self.j2 = self.i2.copy()
        self.e2 = self.i2.copy()
        self.k2 = np.zeros(self.NBars * (4 ** 2))
        index = 0
        for bar in range(self.NBars):
            eDof = np.append([2 * self.Bars[bar].bNode],[2 * self.Bars[bar].bNode + 1], axis = 0).reshape((4,1),order = 'F')
            I = np.repeat(eDof, 4, axis=1)
            J = I.T
            self.i2[index:index + 16] = I.flatten(order = 'F')
            self.j2[index:index + 16] = J.flatten(order = 'F')
            self.e2[index:index + 16] = bar+self.NElem # the bar is after the elements
            index = index + 16
        # ------- System final ------
        NLoad = np.size(self.Load, 0)
        self.F = np.zeros(2 * self.NNode)
        self.F[2 * self.Load[0:NLoad, 0].astype(int)] = self.Load[0:NLoad, 1]
        self.F[(2 * self.Load[0: NLoad, 0].astype(int)) + 1] = self.Load[0: NLoad, 2]
        NSupp = np.size(self.Supp, 0)
        # change because of 0*x = 0
        FixedDofs = np.append((2 * self.Supp[0:NSupp, 0])[self.Supp[0:NSupp, 1].astype(bool)],
                              (2 * self.Supp[0:NSupp, 0] + 1)[self.Supp[0:NSupp, 2].astype(bool)])
        AllDofs = np.arange(0, 2 * self.NNode)
        self.FreeDofs = np.setdiff1d(AllDofs, FixedDofs)
        self.Fl = self.F[self.FreeDofs]

        # assembly k for bars



    def update_k(self):
        bidimensionalUpdate(self.BiDimensionalElements,self.NElem,self.U,self.Node, self.ShapeFnc, self.Nu0,self.Ec, self.Et) # wrapper
        barsUpdate(self.Bars,self.NBars,self.U)
        index = 0
        for el in range(self.NElem):
            NDof = self.ElemNDof[el]
            self.k1[index:index + NDof ** 2] =  self.BiDimensionalElements[el].Ke.flatten(order='F')
            index = index + NDof ** 2

        index = 0
        for b in range(self.NBars):
            self.k2[index:index + 16] = self.Bars[b].Ke.flatten(order='F') * self.Bars[b].BarType
            index = index + 16

        self.k = np.append(self.k1,self.k2)
        self.i = np.append(self.i1,self.i2)
        self.j = np.append(self.j1,self.j2)
        self.e = np.append(self.e1,self.e2)

    def NonLinearFixedPoint(self,E):
        er = np.inf
        tol = 0.005
        inter = 0
        locked = False
        while er>tol:
            erant = er
            Ua = self.U.copy()
            self.update_k()
            self.FEAnalysis(E)
            er = np.mean(abs(self.U - Ua))/np.mean(abs(self.U))
            print(f'\t\t\tFinite Element NonLinear Interaction {inter}  Error = {er}')
            inter = inter +1
            if erant<er and inter>3 and not locked:
                locked = True
                print('\t\t\t\tFixed Point locked, trying deslock')
            if locked == True:
                self.U = (self.U+Ua)/2
                '''correct what a think is a problem for the Fixed Point method
                    in theory is like if the line between the solution and the real graph
                    is trap in a close loop and do not converge this force the estimative go inside 
                    the region of the loop'''




    def FEAnalysis(self, E):
        K = sparse.csr_matrix((self.k * E[self.e], (self.i, self.j)), shape=(max(self.i) + 1, max(self.j) + 1))
        Kl = K[self.FreeDofs,:][:,self.FreeDofs]
        self.U[self.FreeDofs] = spsolve(Kl,self.F[self.FreeDofs])


    #------------------------Tabulate Shape Function--------------------------------------------------------
    def TabShapeFnc(self):
        def PolyTrnglt(nn, xi):
            temp = np.array(range(1,nn+1))
            p = np.append([np.cos(2*np.pi*temp/nn)],[np.sin(2*np.pi*temp/nn)], axis=0).T # triangulation points
            p = np.append(p,[xi], axis=0)
            Tri = np.zeros((nn,3), dtype=int) # connecting the triangles to calculate the barycentric coordinates
            Tri[0:nn,0] = nn
            Tri[0:nn,1] = np.arange(nn)
            Tri[0:nn,2] = np.arange(1,nn+1)
            Tri[nn-1, 2] = 0
            return p, Tri


        def PolyShapeFnc(nn, xi):
            N= np.zeros((nn,1))
            alpha = np.zeros((nn, 1))
            dNdxi = np.zeros((nn, 2))
            dalpha = np.zeros((nn, 2))
            sum_alpha = 0.0
            sum_dalpha = np.zeros((1, 2))
            A = np.zeros((nn, 1))
            dA = np.zeros((nn, 2))
            p, Tri = PolyTrnglt(nn,xi)
            for i in range(0,nn):
                sctr = Tri[i] # vector being analyzed
                pT = p[sctr] # analyzes the points of the triangle
                A[i] = 1 / 2 * np.linalg.det(np.column_stack([pT, np.ones((3, 1))])) #basic area formula
                dA[i, 0] = 1 / 2 * (pT[2, 1] - pT[1, 1]) # derived from the area with respect to xi1
                dA[i, 1] = 1 / 2 * (pT[1, 0] - pT[2, 0]) #derived from the area with respect to xi2
            A = np.append([A[-1,:]],A) # stores the areas
            dA = np.append([dA[-1,:]],dA, axis=0)  # stores derivatives
            for i in range(0,nn):
                alpha[i] = 1 / (A[i] * A[i + 1]) # alphas calculation reduced to regular polygons
                dalpha[i, 0] = -alpha[i] * (dA[i, 0] / A[i] + dA[i + 1, 0] / A[i + 1]) # calculation of alpha derivatives
                dalpha[i, 1] = -alpha[i] * (dA[i, 1] / A[i] + dA[i + 1, 1] / A[i + 1]) # calculation of alpha derivatives
                sum_alpha = sum_alpha + alpha[i] # sum of alphas used to calculate N and dN
                sum_dalpha = sum_dalpha+dalpha[i, :] # sum of the alpha derivatives used to calculate dN
            for i in range(0,nn):
                N[i] = alpha[i]/sum_alpha # Ni calculation
                dNdxi[i,:]=(dalpha[i,:]-N[i]*sum_dalpha)/sum_alpha #dNi calculation
            return N, dNdxi


        def TriQuad():
            point = np.array([[1 / 6, 1 / 6],[2 / 3, 1 / 6],[1 / 6, 2 / 3]])
            weight = np.array([1 / 6, 1 / 6, 1 / 6])
            return weight, point
        def TriShapeFnc(s):
            N = np.array([1 - s[0] - s[1],s[0],s[1]])
            dNds = np.array([[-1,-1],[1,0],[0,1]] )
            return N, dNds

        def PolyQuad(nn):
            W, Q = TriQuad() # quadrature points of the reference triangle
            p, Tri = PolyTrnglt(nn,np.array([0,0])) # regular polygon triangles
            point = np.zeros((nn*len(W),2)) # return
            weight = np.zeros((nn * len(W), 1))
            for k in range(0,nn): #transformed from the triangle to the reference that applies the quadrature
                sctr = Tri[k,:]
                for q in range(0,len(W)):
                    N, dNdS = TriShapeFnc(Q[q,:]) # N is used to find the equivalent square point in the triangulation
                    J0 = p[sctr,:].T @ dNdS # Jacobian to make the area correction factor
                    l = (k) * len(W) + q #store in a vector
                    point[l,:] = N.T @ p[sctr,:] #transforms the square points to the equivalent of the triangle
                    weight[l] = np.linalg.det(J0)*W[q] #weight corrected with the Jacobian determinant
            return weight, point # corresponding points of the polygon

        ElemNNode = [element.size for element in self.BiDimensionalElements]
        self.ShapeFnc = [Generic() for n in range(0,max(ElemNNode)+1)]
        for nn in range(min(ElemNNode),max(ElemNNode)+1): # python open range
            [W,Q] = PolyQuad(nn)
            self.ShapeFnc[nn].W = W
            self.ShapeFnc[nn].N = np.zeros((nn, 1, np.size(W, 0)))
            self.ShapeFnc[nn].dNdxi = np.zeros((nn, 2, np.size(W,0)) )
            for q in range(0,np.size(W,0)):
                N, dNdxi = PolyShapeFnc(nn, Q[q,:]) # calculation of the polygon shape function with reference to the transformed quadrature points
                self.ShapeFnc[nn].N[:,:,q] = N #assign the value
                self.ShapeFnc[nn].dNdxi[:,:, q] = dNdxi # assign the value