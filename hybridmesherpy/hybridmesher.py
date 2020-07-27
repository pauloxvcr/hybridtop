import numpy as np
from polymesherpy.polymesherpy import PolyMesher, PolyMshr_CllpsEdgs
from polymesherpy import domains
from grandpy.generateGsC import generateGS

'''
def PolyTrussMesher(Domain,NElem,MaxIter,Ratio,Lvl,RestrictDomain=None,ColTol = 0.999999):
    Node1,Element1,Supp1,Load1 = PolyMesher(Domain,NElem*Ratio,MaxIter) #micro
    Node2, Element2, Supp2, Load2 = PolyMesher(Domain,NElem,MaxIter) #macro
    Node2,compatib = MesherCompatib(Node1,Node2)
    Node2, Element2 = PolyMshr_CllpsEdgs(Node2,Element2) # verificar se comportamento é o desejado
    Node2,compatib  = MesherCompatib(Node1,Node2)

    Bars = generateGS(Node2,Element2,Lvl,RestrictDomain,ColTol)

    Bars = compatib[Bars] # ver se da erro

    return ResultPolyTrussMesher(Node1, Element1, Supp1, Load1, Bars)'''

def HybridMesher(mesher1,mesher2,Lvl,RestrictDomain=None,ColTol = 0.999999):
    Node1,Element1,Supp1,Load1 = mesher1.get() #micro
    Node2, Element2, Supp2, Load2 = mesher2.get() #macro
    Node2,compatib = MesherCompatib(Node1,Node2)
    Node2, Element2 = PolyMshr_CllpsEdgs(Node2,Element2) # verificar se comportamento é o desejado
    Node2,compatib  = MesherCompatib(Node1,Node2)
    mesher2.Node,mesher2.Element = Node2, Element2
    Bars = generateGS(mesher2,Lvl,RestrictDomain,ColTol)

    Bars = compatib[Bars] # ver se da erro

    return ResultPolyTrussMesher(Node1, Element1, Supp1, Load1, Bars)

def MesherCompatib(Node1,Node2):
    compatib = np.zeros(np.size(Node2,0),dtype=int)
    for j in range(np.size(Node2,0)):
        dist = 10**10
        for i in range(np.size(Node1,0)):
            dist2 = ((Node1[i,0]-Node2[j,0])**2 + (Node1[i,1]-Node2[j,1])**2)**0.5
            if dist2 <=dist:
                compatib[j] = i
                dist = dist2
    Node2 = Node1[compatib,:]
    return Node2, compatib


class ResultPolyTrussMesher():
    def __init__(self,Node,Element,Supp,Load,Bars):
        self.Node = Node
        self.Element = Element
        self.Supp = Supp
        self.Load = Load
        self.Bars = Bars
    def get(self):
        return self.Node,self.Element,self.Supp,self.Load,self.Bars
