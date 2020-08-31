'''
This algorithm is a translation of a MatLab algorithm that can by found with the algorithm article on:
https://paulino.ce.gatech.edu/software.html

Reference of the algorithm article:
Talischi C, Paulino GH, Pereira A, Menezes IFM (2012) PolyMesher: a general-purpose mesh generator for polygonal
elements written in Matlab. Structural and Multidisciplinary Optimization 45:309–328.
https://doi.org/10.1007/s00158-011-0706-z

'''

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import sparse

from . import domains
import time

start = time.time()

def PolyMesher(Domain,NElem,MaxIter,P = None):
    if P is None:
        P=PolyMshr_RndPtSet(NElem,Domain)
    NElem = np.size(P,0) #verificar
    Tol = 5e-6
    It = 0
    Err = 1
    c = 1.5
    BdBox = Domain.BdBox
    PFix = Domain.PFix()
    Area = (BdBox[1] - BdBox[0]) * (BdBox[3] - BdBox[2])
    Pc = P

    while (It <= MaxIter and Err > Tol):
        Alpha = c * np.sqrt(Area / NElem)
        P = Pc
        R_P = PolyMshr_Rflct(P,NElem,Domain,Alpha)
        P, R_P = PolyMshr_FixedPoints(P, R_P, PFix)
        R_P = np.unique(R_P.round(decimals=5), axis=0)# case tolerance for Reflection closes
        vor = Voronoi(np.concatenate((P,R_P),axis=0))
        Node, Element = vor.vertices, np.array(vor.regions)[vor.point_region]  #!!!!!! Verificar diferenças de voronoi
        """ Os nós estão em ordem diferentes"""
        # so pra relatório caso erro
        '''for element in Element[:NElem]:
            for i in element:
                if i == -1:
                    print(f'elemento:{element} i:{i}')
                    #trava para verificar que ocorreu o erro
                    voronoi_plot_2d(vor)
                    plt.show()'''
        Pc,A = PolyMshr_CntrdPly(Element,Node,NElem)
        Area = sum(abs(A))
        Err = np.sqrt(sum((A**2)*np.sum((Pc-P)*(Pc-P),1)))*NElem/(Area**1.5)
        print(f'It: {It}  Error: {Err}')
        It=It+1
        #if NElem <= 2000: PolyMshr_PlotMsh(Node, Element, NElem, fig) #isso deixa lento!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Node, Element = PolyMshr_ExtrNds(NElem, Node, Element) # Extract node list
    Node, Element = PolyMshr_CllpsEdgs(Node, Element, 0.1) # Remove small edges
    Node, Element = PolyMshr_RsqsNds(Node, Element) #Reoder Nodes
    BC = Domain.BC(Node,Element)
    Supp = BC[0]
    Load = BC[1]
    PolyMshr_PlotMsh(Node, Element, Supp,Load)
    return ResultPolyMesher(Node,Element,Supp,Load)



def PolyMshr_RndPtSet(NElem,Domain):
    P = np.zeros((NElem,2))
    BdBox = Domain.BdBox
    Ctr = 0
    while Ctr<NElem:
        Y1 = (BdBox[1]-BdBox[0])*np.random.rand(NElem,1)+BdBox[0]
        Y2 = (BdBox[3]-BdBox[2])*np.random.rand(NElem,1)+BdBox[2]
        Y = np.concatenate((Y1,Y2),axis=1)
        d = Domain.Dist(Y)
        I = np.argwhere(d[:,-1]<0).flatten() #Index of seeds inside the domain
        NumAdded = min(NElem-Ctr,len(I))
        P[Ctr: Ctr + NumAdded,:] = Y[I[0: NumAdded],:]
        Ctr = Ctr + NumAdded
    return P


def PolyMshr_Rflct(P,NElem,Domain,Alpha):
    eps = 1e-8
    eta = 0.9
    d = Domain.Dist(P)
    NBdrySegs = np.size(d, 1) - 1 # Number of constituent bdry segments
    n1 = (Domain.Dist(P+np.repeat(np.array([[eps,0]]),NElem,0))-d)/eps #verificar se vai avaliar como matriz [eps,0]
    n2 = (Domain.Dist(P + np.repeat(np.array([[0, eps]]), NElem, 0))-d)/eps
    I = np.nonzero(abs(d[:, 0: NBdrySegs]) < Alpha) # Logical index of seeds near the bdry modificação/ brusca verificar se é matriz ou vetor(a saida é uma tupla)
    #é tupla pois deve fazer a combinação no vetor eg x[1,2]
    P1 = np.repeat(np.atleast_2d(P[:,0]),NBdrySegs+1,axis=0).T #[NElem x NBdrySegs] extension of P(:,1) # a função de distancia possui 4+1(max) valores
    P2 = np.repeat(np.atleast_2d(P[:,1]),NBdrySegs+1,axis=0).T #[NElem x NBdrySegs] extension of P(:,2)

    R_P = P1[I] - 2 * n1[I] * d[I] # calcula as reflexões somente dos pertos do contorno
    R_P = np.column_stack([R_P,P2[I] - 2 * n2[I] * d[I] ]) #y das reflexões a partir daqui vira novamente pontos

    d_R_P = Domain.Dist(R_P) #calcula a função de distancia das reflexões
    J, = np.nonzero(np.logical_and(abs(d_R_P[:,-1])>=eta*abs(d[I]),d_R_P[:,-1]>0)) #em domínio não convexos o reflexo pode ta mais proximo de um dos contornos do que a semente em relação ao contorno de reflexão, isso atrapalharia. a segunda condição é que tem que ta externa ao domínio
    R_P = R_P[J,:]
    R_P = np.unique(R_P,axis=0)
    return R_P


def PolyMshr_FixedPoints(P, R_P, PFix):
    PP = np.concatenate((P,R_P),axis=0)
    for i in range(np.size(PFix,0)):
        vector= np.sqrt((PP[:,0]-PFix[i,0])**2+(PP[:,1]-PFix[i,1])**2)
        I = np.argsort(vector)
        B = np.take_along_axis(vector, I, axis=0)
        for j in range(1,4):
            n = PP[I[j],:] - PFix[i,:]
            n= n/np.linalg.norm(n)
            PP[I[j],:] = PP[I[j],:]-n * (B[j] - B[0])

    P = PP[0:np.size(P, 0), :]
    R_P = PP[(np.size(P, 0)):len(PP), :] #correção de erro que tava cortando uma reflexão
    return P, R_P



def PolyMshr_CntrdPly(Element,Node,NElem):
    Pc =np.zeros((NElem,2))
    A = np.zeros(NElem)
    for el in range(NElem):
        vx = Node[Element[el],0]
        vy = Node[Element[el],1]
        nv = len(Element[el])
        vxS = np.append(vx[1:nv],vx[0]) #precisa inverter elementos
        vyS = np.append(vy[1:nv], vy[0])
        temp = vx*vyS - vy*vxS
        A[el] = 0.5 * sum(temp)
        Pc[el,:] = 1/(6*A[el])*np.array([sum((vx+vxS)*temp), sum((vy+vyS)*temp)])
    return Pc , A

def PolyMshr_ExtrNds(NElem, Node0, Element0):
    #temp for flaten
    #cria um vetor com todos os nós da malha
    Element0Flat = []
    for item in Element0[0:NElem]:
        for i in item:
            Element0Flat.append(i)
    map = np.unique(np.array(Element0Flat)) # o vetor anterior tem repetições, precisa tirá-las
    cNode = np.array(range(0,np.size(Node0,0))) # vetor cNode que será usada para fazer o mapeamento de mudança
    cNode[np.setdiff1d(cNode,map)] = max(map) # seleciona os nós externos que serão retirados e atribui o valor max(map)
    Node, Element = PolyMshr_RbldLists(Node0,Element0[0:NElem],cNode) #observe que usa somente a parte de elementos referente as sementes
    return Node, Element

def PolyMshr_RbldLists(Node0,Element0,cNode):
    Element = []
    foo, ix, jx = np.unique(cNode,return_index=True,return_inverse=True) #cNode[ix]=foo e foo[jx] = cNode
    '''if np.size(jx) != np.size(cNode): #desnecessário
        jx = jx.transpose()'''
    if np.size(Node0,0)>len(ix): # aparentemente poderia se utilizar foo, manter so na duvida(ix está em ordem desta forma o ultimo valor será o de um no retirado)
        ix[-1] = max(cNode)
    Node = Node0[ix,:] # faz as modificações necessárias e retira os nos desnecessários
    for el in range(0,np.size(Element0,0)): #precisa agora fazer a transformação nos elementos
        Element.append(np.unique(jx[Element0[el]]))# no colapso pode ter repetição de nós(que serão colapsados) / jx é como uma inversa degenerada mas serve adequadamente
        vx = Node[Element[el],0] # a partir daqui será colocado em ordem antihoraria
        vy = Node[Element[el],1] # coordenadas dos nós
        nv = len(vx)
        vector = np.arctan2(vy-sum(vy)/nv,vx-sum(vx)/nv) #o arctan determinará a ordem
        iix = np.argsort(vector) #
        # foo = np.take_along_axis(vector, iix, axis=0) linha desnecessária
        Element[el] = Element[el][iix] #coloca em ordem antihoraria(importante para avaliar beta)
        #observar sort e unique no numpy
    return Node, Element

def PolyMshr_CllpsEdgs(Node0,Element0,Tol=0.1):
    """ Colapsa os nós"""
    while True:
        cEdge = np.empty((0,2),dtype=int)
        for el in range(0,len(Element0)):
            if np.size(Element0[el])<4: continue # não pode colapsar triangulos
            vx = Node0[Element0[el],0]  # coordenadas
            vy = Node0[Element0[el],1]
            nv = len(vx) #numero de nós nos poligonos
            beta = np.arctan2(vy-sum(vy)/nv, vx-sum(vx)/nv) # sum(vx)/nv e sum(vy)/nv são para botar em referencia ao centro do pol então ai é o angulo total do ponto
            beta = np.remainder(np.append(beta[1:nv],beta[0]) - beta,2*np.pi) #subtrai o angulo dos pontos adjacentes para obter o angulo do lado
            betaIdeal = 2*np.pi/nv  # valor a que será comparado
            Edge = np.stack([Element0[el],np.append(Element0[el][1:nv],Element0[el][0])]).T
            cEdge = np.append(cEdge,Edge[beta<Tol*betaIdeal], axis=0)#pega somente os lados que tem a condição de colapso

        if (np.size(cEdge,0)==0): break # essa parte é pq é desnecessário o resto se não tem lado para colapsar
        cEdge = np.unique(np.sort(cEdge,1),axis=0) #no mínimo serão dois poligonos envolvidos então tem que tirar repetição
        cNode = np.array(range(0,np.size(Node0,0))) # criar o vetor cnode de mapeamento
        for i in range(0,np.size(cEdge,0)):  cNode[cEdge[i,1]] = cNode[cEdge[i,0]] # os nós dos lados que entrarão em colapso são mapiados para o mesmo
        Node0,Element0 = PolyMshr_RbldLists(Node0,Element0,cNode) # usa a função Rbld para fazer a parada acima
    return Node0,Element0

def PolyMshr_RsqsNds(Node0, Element0):
    NNode0 = np.size(Node0, 0)
    NElem0 = len(Element0)
    ElemLnght = np.array([len(element) for element in Element0])
    nn = sum(ElemLnght ** 2)
    i = np.zeros(nn)
    j = np.zeros(nn)
    s = np.zeros(nn)
    index = 0
    for el in range(0, NElem0):
        eNode = Element0[el]
        ElemSet = np.array(range(index, index + ElemLnght[el] ** 2))
        i[ElemSet] = np.kron(eNode, np.ones((ElemLnght[el], 1))).flatten()
        j[ElemSet] = np.kron(eNode, np.ones((1, ElemLnght[el]))).flatten()
        s[ElemSet] = 1
        index = index + ElemLnght[el] ** 2
    K = sparse.coo_matrix((s, (i, j)), shape=(NNode0, NNode0)).tocsr()
    p = sparse.csgraph.reverse_cuthill_mckee(K)
    cNode = np.zeros(NNode0)
    cNode[p[0:NNode0]] = np.array(range(0, NNode0))  # Loucuras
    Node, Element = PolyMshr_RbldLists(Node0, Element0, cNode)
    return Node, Element

def PolyMshr_PlotMsh(Node, Element, Supp= None, Load=None, ):
    fig, ax = plt.subplots()
    patches = []
    num_polygons = len(Element)
    for i in range(num_polygons):
        polygon = Polygon(Node[Element[i]])
        patches.append(polygon)
    p = PatchCollection(patches,edgecolors=('black',),facecolors='white')
    ax.add_collection(p)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.axis('off')
#classe para salvar malha
class ResultPolyMesher():
    def __init__(self,Node,Element,Supp,Load):
        self.Node = Node
        self.Element = Element
        self.Supp = Supp
        self.Load = Load
    def get(self):
        return self.Node,self.Element,self.Supp,self.Load



