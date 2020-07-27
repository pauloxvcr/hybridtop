import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from grandpy.GRANDpy import PlotBoundary



class Generic():
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
        self.W = None
        self.N = None
        self.dNdxi = None

class PolyTop:
    def __init__(self, VolFrac : float,  P,MatIntFnc,zIni = None,zMin=0.0,zMax=1.0,Tol = 0.01, MaxIter = 150, OCMove = 0.2, OCEta = 0.5, penal = 1.0):
        self.VolFrac = VolFrac
        self.P = P
        if zIni == None:
            zIni = VolFrac*np.ones(np.size(P,1))
        self.z = zIni
        self.MatIntFnc = MatIntFnc
        self.zMin=zMin
        self.zMax=zMax
        self.Tol = Tol
        self.MaxIter = MaxIter
        self.OCMove = OCMove
        self.OCEta = OCEta
        self.penal = penal


    def set_fem(self,fem):
        self.fem = fem

    def Opt(self):
        Iter = 0
        Tol = self.Tol * (self.zMax - self.zMin)
        Change = 2 * Tol
        z = self.z
        P = self.P
        try: fem = self.fem
        except: raise RuntimeError("you don't defined the finite element yet")
        E,dEdy,V,dVdy = self.MatIntFnc((P @ z).flatten(),self.penal) #verificar a multiplicação
        # FigHandle, FigData = InitialPlot(fem,V)
        while(Iter<self.MaxIter) and (Change>Tol):
            Iter = Iter+1
            #Compute cost functionals and analysis sensitivities
            f,dfdE,dfdV = self.ObjectiveFnc(E,V)
            g,dgdE,dgdV = self.ConstraintFnc(E,V)
            #Compute design sensitivities
            dfdz = P.transpose() @ (dEdy*dfdE + dVdy*dfdV)
            dgdz = P.transpose() @ (dEdy*dgdE + dVdy*dgdV)
            #Update design variable and analysis parameters
            z,Change = self.UpdateScheme(dfdz,g,dgdz,z)
            E,dEdy,V,dVdy = self.MatIntFnc((P @ z).flatten(), self.penal)
            print(f'It: {Iter} \t Objective: {f} \tChange: {Change}')
        self.z = z
        self.V = V

    def ObjectiveFnc(self,E,V):
        fem = self.fem
        U = fem.FEAnalysis(E)
        f = np.dot(fem.F,U)
        temp = np.cumsum(-U[fem.i]* fem.k* U[fem.j])
        temp = temp[np.cumsum(fem.ElemNDof**2)-1]
        dfdE = np.append(temp[0],temp[1:len(temp)]-temp[0:-1])
        dfdV = np.zeros(np.size(V))
        return f,dfdE,dfdV

    def ConstraintFnc(self,E,V):
        fem = self.fem
        if not hasattr(fem,'ElemArea'):
            fem.setElemArea()
        g = sum(fem.ElemArea*V)/sum(fem.ElemArea) - self.VolFrac
        dgdE = np.zeros(np.size(E))
        dgdV = fem.ElemArea/sum(fem.ElemArea)
        return g,dgdE,dgdV

    def UpdateScheme(self,dfdz,g,dgdz,z0): # ver substitui por uma função scypy
        zMin = self.zMin
        zMax = self.zMax
        move = self.OCMove*(zMax-zMin)
        eta = self.OCEta
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
        return zNew,Change

    def polyPlot(self):
        fig, ax = plt.subplots()
        Element = self.fem.Element
        num_polygons = self.fem.NElem
        c = 1 - self.V
        c[c<0] = 0
        for i in range(num_polygons):

            polygon = Polygon(self.fem.Node[Element[i]],color=[c[i],c[i],c[i]])
            ax.add_patch(polygon)
        ax.autoscale()
        ax.set_aspect('equal')
        ax.axis('off')
        PlotBoundary(self.fem.Node,Element,ax)
    def save_result(self):
        return ResultPolyTop(self.fem.Mesher,self.V,self.z)
# -------------------------------------- Finite Element ----------------------------------------------------------------
class FiniteElement:
    def __init__(self, Mesher, Nu0 = 0.3, E0=2000.0, Reg = 0):
        self.Mesher = Mesher
        self.Node = self.Mesher.Node
        self.Element = self.Mesher.Element #VERIFICAR SE ELEMENT VEM COMO ARRAY
        self.Supp = self.Mesher.Supp
        self.Load = self.Mesher.Load
        self.NNode = np.size(self.Node,0)
        self.NElem = np.size(self.Element, 0)
        self.Nu0 = Nu0
        self.E0 = E0
        self.Reg = Reg
        self.setElemArea() #!!

    def FEAnalysis(self,E):
        if not hasattr(self,'k'):
            self.ElemNDof = 2*np.array([len(element) for element in self.Element])
            self.i = np.zeros(sum(self.ElemNDof ** 2),dtype=int)
            self.j = self.i.copy()
            self.e = self.i.copy()
            self.k = np.zeros(sum(self.ElemNDof ** 2))
            index = 0
            if not hasattr(self, 'ShapeFnc'):
                self.TabShapeFnc()
            for el in range(0,self.NElem):
                Ke = self.LocalK(self.Element[el])
                NDof = self.ElemNDof[el]
                eDof = np.append([2*self.Element[el]],[2*self.Element[el]+1],axis=0).reshape((NDof,1),order='F')
                I = np.repeat(eDof,NDof,axis=1)
                J = I.transpose()
                self.i[index:index+NDof**2] = I.flatten(order='F')
                self.j[index:index+NDof**2] = J.flatten(order='F')
                self.k[index:index+NDof**2] = Ke.flatten(order='F')
                self.e[index:index+NDof**2] = el
                index = index + NDof**2 #Verificar questão do intervalo aberto
        NLoad = np.size(self.Load,0)
        self.F  = np.zeros(2*self.NNode)
        self.F[2*self.Load[0:NLoad,0].astype(int)]= self.Load[0:NLoad,1]
        self.F[(2 * self.Load[0: NLoad,0].astype(int))+1]   = self.Load[0: NLoad, 2]
        NSupp = np.size(self.Supp,0)
        #Alteração devido ao erro de 0*x = 0
        FixedDofs = np.append((2 * self.Supp[0:NSupp, 0])[self.Supp[0:NSupp, 1].astype(bool)],
                              (2 * self.Supp[0:NSupp, 0] + 1)[self.Supp[0:NSupp, 2].astype(bool)])
        AllDofs = np.arange(0,2*self.NNode)
        self.FreeDofs = np.setdiff1d(AllDofs, FixedDofs)
        K = sparse.csr_matrix((self.k * E[self.e].flatten(), (self.i, self.j)), shape=(max(self.i) + 1, max(self.j) + 1))
        U = np.zeros(2*self.NNode)
        U[self.FreeDofs] = spsolve(K[self.FreeDofs,:][:,self.FreeDofs],self.F[self.FreeDofs]) # olhar para assumir que é simetrica
        return U


    def setElemArea(self):
        self.ElemArea = np.zeros(self.NElem)
        for el in range(0,self.NElem):
            vx = self.Node[self.Element[el],0]
            vy = self.Node[self.Element[el],1]
            self.ElemArea[el] = 0.5 * sum(vx * np.append(vy[1:len(vy)],vy[0])-vy * np.append(vx[1:len(vx)],vx[0]))


    def LocalK(self,eNode):
        D = self.E0/(1-self.Nu0**2)*np.array([[1,self.Nu0,0],[self.Nu0,1,0],[0,0,(1-self.Nu0)/2]])
        nn = len(eNode)
        Ke=np.zeros((2*nn,2*nn))
        W = self.ShapeFnc[nn].W
        for q in range(0,len(W)):
            dNdxi = self.ShapeFnc[nn].dNdxi[:,:,q]
            J0 = self.Node[eNode,:].T @ dNdxi
            dNdx =dNdxi @ np.linalg.inv(J0) # verificar melhor forma
            B = np.zeros((3,2*nn))
            B[0, 0: 2 * nn:2] = dNdx[:, 0]
            B[1, 1: 2 * nn:2] = dNdx[:, 1]
            B[2, 0: 2 * nn: 2] = dNdx[:, 1]
            B[2, 1: 2 * nn: 2] = dNdx[:, 0]
            Ke = Ke +B.transpose() @ D @ B * W[q] * np.linalg.det(J0) # verificar se W é escalar
        return Ke


    def TabShapeFnc(self):
        def PolyTrnglt(nn, xi):
            temp = np.array(range(1,nn+1))
            p = np.append([np.cos(2*np.pi*temp/nn)],[np.sin(2*np.pi*temp/nn)], axis=0).T # são os pontos da triangulação
            p = np.append(p,[xi], axis=0)
            Tri = np.zeros((nn,3), dtype=int) # é a conexão dos triangulos para calcular as coordenadas baricêntricas
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
                sctr = Tri[i] # significa o vetor sendo analisado
                pT = p[sctr] # analisa os pontos do triangulo
                A[i] = 1 / 2 * np.linalg.det(np.column_stack([pT, np.ones((3, 1))])) #formula básica de área
                dA[i, 0] = 1 / 2 * (pT[2, 1] - pT[1, 1]) # derivada da área com relação a xi1
                dA[i, 1] = 1 / 2 * (pT[1, 0] - pT[2, 0]) #derivada da área com relação a x2
            A = np.append([A[-1,:]],A) # armazena as áreas
            dA = np.append([dA[-1,:]],dA, axis=0)  # armazena as derivadas
            for i in range(0,nn):
                alpha[i] = 1 / (A[i] * A[i + 1]) # calculo dos alfas reduzido para regular
                dalpha[i, 0] = -alpha[i] * (dA[i, 0] / A[i] + dA[i + 1, 0] / A[i + 1]) # calculo das derivadas de alpha
                dalpha[i, 1] = -alpha[i] * (dA[i, 1] / A[i] + dA[i + 1, 1] / A[i + 1]) #calculo das derivadas de alpha
                sum_alpha = sum_alpha + alpha[i] #soma dos alphas utilizado para calcular N e dN
                sum_dalpha = sum_dalpha+dalpha[i, :] # soma das derivadas de alpha usada para calcular dN
            for i in range(0,nn):
                N[i] = alpha[i]/sum_alpha # calculo dos Ni
                dNdxi[i,:]=(dalpha[i,:]-N[i]*sum_dalpha)/sum_alpha #Calculo dos dNi
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
            W, Q = TriQuad() # pontos de quadratura do triangulo de referência
            p, Tri = PolyTrnglt(nn,np.array([0,0])) # triangulos do poligono regular
            point = np.zeros((nn*len(W),2)) #saída
            weight = np.zeros((nn * len(W), 1))
            for k in range(0,nn): # faz a transformada do triangulo para o de referência que aplica a quadratura
                sctr = Tri[k,:]
                for q in range(0,len(W)):
                    N, dNdS = TriShapeFnc(Q[q,:]) # o N é usado para encontrar o ponto equivalente da quadratura na triangulação
                    J0 = p[sctr,:].T @ dNdS # jacobiano para fazer o fator de correção da área
                    l = (k) * len(W) + q #somente para armazenar em um vetor
                    point[l,:] = N.T @ p[sctr,:] #transforma os pontos da quadratura para o equivalente ao triangulo
                    weight[l] = np.linalg.det(J0)*W[q] #peso corrigido com o determinante do jacobiano
            return weight, point # saira os pontos correspondetes do poligono ja

        ElemNNode = [len(element) for element in self.Element]
        self.ShapeFnc = [Generic() for n in range(0,max(ElemNNode)+1)]
        for nn in range(min(ElemNNode),max(ElemNNode)+1): # devido ao intervalo aberto do python
            [W,Q] = PolyQuad(nn)
            self.ShapeFnc[nn].W = W
            self.ShapeFnc[nn].N = np.zeros((nn, 1, np.size(W, 0)))
            self.ShapeFnc[nn].dNdxi = np.zeros((nn, 2, np.size(W,0)) )
            for q in range(0,np.size(W,0)):
                N, dNdxi = PolyShapeFnc(nn, Q[q,:]) # aqui será calculado a função de forma do poligono com referencia aos pontos da quadratura transformados
                self.ShapeFnc[nn].N[:,:,q] = N #atribui o valor
                self.ShapeFnc[nn].dNdxi[:,:, q] = dNdxi # atribui o valor

#----------------------------- Result ----------------------------------------------------------------------------------
class ResultPolyTop:
    def __init__(self, Mesher,V,z):
        self.Mesher = Mesher
        self.V = V
        self.z = z
def plotResult(Result):
    fig, ax = plt.subplots()
    Element = Result.Mesher.Element
    num_polygons = len(Result.Mesher.Element)
    c = 1 - Result.V
    c[c < 0] = 0
    for i in range(num_polygons):
        polygon = Polygon(Result.Mesher.Node[Element[i]], color=[c[i], c[i], c[i]])
        ax.add_patch(polygon)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.axis('off')


























