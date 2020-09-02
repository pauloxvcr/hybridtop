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
        Node, Element = vor.vertices, np.array(vor.regions)[vor.point_region] # put the nodes in order (in matlab there is not need for this)
        # if a intern seed for any reason generate a cell that has no limit(going to infinite) the program do not work proper
        # this is rare but if happens you can debug with this
        '''for element in Element[:NElem]:
            for i in element:
                if i == -1:
                    print(f'elemento:{element} i:{i}')
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
    n1 = (Domain.Dist(P+np.repeat(np.array([[eps,0]]),NElem,0))-d)/eps
    n2 = (Domain.Dist(P + np.repeat(np.array([[0, eps]]), NElem, 0))-d)/eps
    I = np.nonzero(abs(d[:, 0: NBdrySegs]) < Alpha) # Logical index of seeds near the bdry modificação/ brusca verificar se é matriz ou vetor(a saida é uma tupla)
    #tupla pois deve fazer a combinação no vetor eg x[1,2]
    P1 = np.repeat(np.atleast_2d(P[:,0]),NBdrySegs+1,axis=0).T # [NElem x NBdrySegs] extension of P(:,1) # a função de distancia possui 4+1(max) valores
    P2 = np.repeat(np.atleast_2d(P[:,1]),NBdrySegs+1,axis=0).T # [NElem x NBdrySegs] extension of P(:,2)

    R_P = P1[I] - 2 * n1[I] * d[I] # Calculates the reflections only close to the boundary
    R_P = np.column_stack([R_P,P2[I] - 2 * n2[I] * d[I] ]) # insert the y coordinate of the reflections

    d_R_P = Domain.Dist(R_P) # sign function of the reflections
    J, = np.nonzero(np.logical_and(abs(d_R_P[:,-1])>=eta*abs(d[I]),d_R_P[:,-1]>0)) # in non-convex domains the reflection may be closer to one of the contours than the seed in relation to the reflection contour, this would hinder. the second condition is: It must be external to the domain
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
    R_P = PP[(np.size(P, 0)):len(PP), :]
    return P, R_P



def PolyMshr_CntrdPly(Element,Node,NElem):
    Pc =np.zeros((NElem,2))
    A = np.zeros(NElem)
    for el in range(NElem):
        vx = Node[Element[el],0]
        vy = Node[Element[el],1]
        nv = len(Element[el])
        vxS = np.append(vx[1:nv],vx[0]) #
        vyS = np.append(vy[1:nv], vy[0])
        temp = vx*vyS - vy*vxS
        A[el] = 0.5 * sum(temp)
        Pc[el,:] = 1/(6*A[el])*np.array([sum((vx+vxS)*temp), sum((vy+vyS)*temp)])
    return Pc , A

def PolyMshr_ExtrNds(NElem, Node0, Element0):
    #temp for flaten
    #creat a vector with all the nodes
    Element0Flat = []
    for item in Element0[0:NElem]:
        for i in item:
            Element0Flat.append(i)
    map = np.unique(np.array(Element0Flat)) # remove repetitions
    cNode = np.array(range(0,np.size(Node0,0))) # vector cNode that will be used for map the change
    cNode[np.setdiff1d(cNode,map)] = max(map) # Select the external nodes that will be removed and assign the value max(map)
    Node, Element = PolyMshr_RbldLists(Node0,Element0[0:NElem],cNode) # only uses the elements referents to the seeds
    return Node, Element

def PolyMshr_RbldLists(Node0,Element0,cNode):
    Element = []
    foo, ix, jx = np.unique(cNode,return_index=True,return_inverse=True) # cNode[ix]=foo e foo[jx] = cNode
    '''if np.size(jx) != np.size(cNode): #desnecessário
        jx = jx.transpose()'''
    if np.size(Node0,0)>len(ix): #
        ix[-1] = max(cNode)
    Node = Node0[ix,:] # makes necessary modifications and removes unnecessary nodes
    for el in range(0,np.size(Element0,0)): # now needs to make the transformation in the elements
        Element.append(np.unique(jx[Element0[el]]))# in the collapse there may be repetition of nodes (which will be collapsed) / jx is like a degenerate inverse but it serves properly
        vx = Node[Element[el],0] # will be placed in anti-hourly order
        vy = Node[Element[el],1] # node coordinates
        nv = len(vx)
        vector = np.arctan2(vy-sum(vy)/nv,vx-sum(vx)/nv) # the arctan will determine the order
        iix = np.argsort(vector) #
        # foo = np.take_along_axis(vector, iix, axis=0) unnecessary line
        Element[el] = Element[el][iix] # puts in order anti-clockwise (important to evaluate beta)

    return Node, Element

def PolyMshr_CllpsEdgs(Node0,Element0,Tol=0.1):
    """ Colapsa os nós"""
    while True:
        cEdge = np.empty((0,2),dtype=int)
        for el in range(0,len(Element0)):
            if np.size(Element0[el])<4: continue # cannot collapse triangles
            vx = Node0[Element0[el],0]  # coordinates
            vy = Node0[Element0[el],1]
            nv = len(vx) # number of nodes in the polygons
            beta = np.arctan2(vy-sum(vy)/nv, vx-sum(vx)/nv) # sum(vx)/nv e sum(vy)/nv establishes reference to the center of the pol so this line is the total angle of the point
            beta = np.remainder(np.append(beta[1:nv],beta[0]) - beta,2*np.pi) # subtract the angle of the adjacent points to obtain the angle of the side
            betaIdeal = 2*np.pi/nv  # value that will be compared
            Edge = np.stack([Element0[el],np.append(Element0[el][1:nv],Element0[el][0])]).T
            cEdge = np.append(cEdge,Edge[beta<Tol*betaIdeal], axis=0)# selects only the sides that have the collapse condition

        if (np.size(cEdge,0)==0): break # unnecessary what is below if there is no side to collapse
        cEdge = np.unique(np.sort(cEdge,1),axis=0) # at least there will be two polygons involved so you have to remove repetition
        cNode = np.array(range(0,np.size(Node0,0))) # create the mapping cnode vector
        for i in range(0,np.size(cEdge,0)):  cNode[cEdge[i,1]] = cNode[cEdge[i,0]] # the collapsing side nodes are mapped to the same
        Node0,Element0 = PolyMshr_RbldLists(Node0,Element0,cNode) # uses the Rbld function to do the above job
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
    cNode[p[0:NNode0]] = np.array(range(0, NNode0))
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
#Class to save the result
class ResultPolyMesher():
    def __init__(self,Node,Element,Supp,Load):
        self.Node = Node
        self.Element = Element
        self.Supp = Supp
        self.Load = Load
    def get(self):
        return self.Node,self.Element,self.Supp,self.Load



