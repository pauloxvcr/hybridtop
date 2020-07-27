import numpy as np
from .distFunctions import *

class GenericDomain():
    def Dist(self,P):
        pass
    def BC(self,Node,Element):
        pass
    def PFix(self):
        pass


class MbbDomain(GenericDomain):
    def __init__(self):
        self.BdBox = np.array([0,2,0,1])

    def Dist(self, P):
        return dRectangle(P, self.BdBox[0], self.BdBox[1], self.BdBox[2], self.BdBox[3])

    def BC(self,Node,Element=None):
        eps = 0.1 * np.sqrt((self.BdBox[1] - self.BdBox[0]) * (self.BdBox[3] - self.BdBox[2]) / np.size(Node, 0))
        LeftEdgeNodes = np.nonzero(abs(Node[:, 0]-self.BdBox[0]) < eps)
        LeftUpperNode = np.nonzero(np.logical_and(abs(Node[:, 0]-self.BdBox[0]) < eps , abs(Node[:, 1]-self.BdBox[3]) < eps))
        RigthBottomNode = np.nonzero(np.logical_and(abs(Node[:, 0]-self.BdBox[1]) < eps , abs(Node[:, 1]-self.BdBox[2]) < eps))
        FixedNodes = np.append(LeftEdgeNodes,RigthBottomNode)
        Supp = np.zeros((len(FixedNodes), 3),dtype=int)
        Supp[:, 0] = FixedNodes
        Supp[0: -1, 1]=1
        Supp[-1, 2] = 1
        Load = np.append(LeftUpperNode,[0,-0.5]).reshape(1,3)
        return Supp, Load

    def PFix(self):
        return []

class MichellDomain(GenericDomain):
    def __init__(self):
        self.BdBox = np.array([0,5,-2,2])
    def Dist(self,P):
        d1 = dRectangle(P, self.BdBox[0], self.BdBox[1], self.BdBox[2], self.BdBox[3])
        d2 = dCircle(P, 0, 0, self.BdBox[3] / 2)
        return dDiff(d1, d2)
    def BC(self, Node, Element =None):
        eps = 0.1 * np.sqrt((self.BdBox[1] - self.BdBox[0]) * (self.BdBox[3] - self.BdBox[2]) / np.size(Node, 0))
        CircleNodes, = np.nonzero(abs(np.sqrt(Node[:,0]**2+Node[:,1]**2)-1.0)<eps)
        Supp = np.ones((len(CircleNodes), 3))
        Supp[:, 0] = CircleNodes
        MidRightFace = np.sqrt((Node[:, 0] - self.BdBox[1])** 2 + (Node[:, 1]-(self.BdBox[2] + self.BdBox[3]) / 2)** 2)
        MidRightFace = np.argsort(MidRightFace)
        Load = np.array([[MidRightFace[0],0,-1]])
        return Supp, Load
    def PFix(self):
        return np.array([[5,0]])

class HPierDomain(GenericDomain):
    def __init__(self):
        self.BdBox = np.array([0,1/2,0,3/4])
        self.RectD1 = np.array([0,1/12,0,3/4])
        self.RectD2 = np.array([1/12 ,1/2 ,1/2 ,3/4])
        self.Line = np.array([1/12,1/2,1/2,23/36])
    def Dist(self,P):
        d1 = dRectangle(P,self.RectD1[0],self.RectD1[1],self.RectD1[2],self.RectD1[3])
        d2 = dRectangle(P,self.RectD2[0],self.RectD2[1],self.RectD2[2],self.RectD2[3])
        d3 = dLine(P,self.Line[0],self.Line[1],self.Line[2],self.Line[3])
        d4 = dDIntesect(d2,d3)
        return dUnion(d4,d1)
    def BC(self,Node,Element):
        eps = 0.1 * np.sqrt((self.BdBox[1] - self.BdBox[0]) * (self.BdBox[3] - self.BdBox[2]) / np.size(Node, 0))
        LeftEdgeNodes, = np.nonzero(abs(Node[:, 0]-self.BdBox[0]) < eps)
        BottomNodes, = np.nonzero(abs(Node[:,1]-self.BdBox[2])<eps)
        LoadNode1 = np.nonzero(np.logical_and(abs(Node[:, 0]-self.BdBox[1]) < eps , abs(Node[:, 1]-self.BdBox[3]) < eps))[0][0]
        LoadNode2 = np.nonzero(np.logical_and(abs(Node[:, 0]-(self.BdBox[1]/3)) < eps , abs(Node[:, 1]-(self.BdBox[3])) < eps))[0][0]
        FixedNodes = np.append(LeftEdgeNodes,BottomNodes)
        Supp = np.zeros((len(FixedNodes), 3), dtype=int)
        Supp[:, 0] = FixedNodes
        Supp[0: len(Supp), 1] = 1
        Supp[len(LeftEdgeNodes):len(Supp), 2] = 1
        Load = np.array([[LoadNode1,0, -0.5],
                         [LoadNode2,0,-0.5]])

        return Supp, Load
    def PFix(self):
        return np.array([[ self.BdBox[1]/3 , self.BdBox[3] ]])

class ShearWallFibDomain(GenericDomain):
    def __init__(self):
        self.BdBox = np.array([0, 8.1,0,8.1])
        self.RectD1 = [3.600,4.500,0,1.800]
        self.RectD2 = [3.600,4.500,2.700,4.5]
        self.RectD3 = [3.600,4.500,5.4,7.2]
    def Dist(self,P):
        d1 = dRectangle(P, self.BdBox[0], self.BdBox[1], self.BdBox[2], self.BdBox[3])
        d2 = dRectangle(P,self.RectD1[0], self.RectD1[1], self.RectD1[2], self.RectD1[3])
        d3 = dRectangle(P,self.RectD2[0], self.RectD2[1], self.RectD2[2], self.RectD2[3])
        d4 = dRectangle(P,self.RectD3[0], self.RectD3[1], self.RectD3[2], self.RectD3[3])
        d = dDiff(d1,d2)
        d = dDiff(d,d3)
        d = dDiff(d,d4)
        return d
    def BC(self,Node,Element):
        eps = 0.1 * np.sqrt((self.BdBox[1] - self.BdBox[0]) * (self.BdBox[3] - self.BdBox[2]) / np.size(Node, 0))
        BottomNodes, = np.nonzero(abs(Node[:, 1] - self.BdBox[2]) < eps)
        LeftNode1, = np.nonzero(np.logical_and(abs(Node[:, 0]-self.BdBox[0]) < eps , abs(Node[:, 1]-self.BdBox[3]) < eps))
        LeftNode2, = np.nonzero(np.logical_and(abs(Node[:, 0]-self.BdBox[0]) < eps , abs(Node[:, 1]-(self.BdBox[3]-2.7)) < eps))
        LeftNode3, = np.nonzero(np.logical_and(abs(Node[:, 0]-self.BdBox[0]) < eps , abs(Node[:, 1]-(self.BdBox[3]-5.4)) < eps))
        Supp = np.ones((len(BottomNodes),3))
        Supp[:,0] = BottomNodes
        Load = np.array([[LeftNode1[0],1,0],
                         [LeftNode2[0], 1, 0],
                         [LeftNode3[0], 1, 0]])
        return Supp,Load
    def PFix(self):
        return np.array([[self.BdBox[0],self.BdBox[3]],
                         [self.BdBox[0],self.BdBox[3]-2.7],
                         [self.BdBox[0], self.BdBox[3]-5.4]
                    ])

class LshapeDomain(GenericDomain):
    def __init__(self):
        self.BdBox = np.array([0.0, 1.4, 0, 0.6])
        self.Rectangle1 = np.array([0.3,0.7,0.2,0.4])
        self.Rectangle2 = np.array([1.0,1.4,0.3,0.6])
    def Dist(self,P):
        d1 = dRectangle(P,self.BdBox[0],self.BdBox[1],self.BdBox[2],self.BdBox[3])
        d2 = dRectangle(P,self.Rectangle1[0],self.Rectangle1[1],self.Rectangle1[2],self.Rectangle1[3])
        d3 = dRectangle(P,self.Rectangle2[0],self.Rectangle2[1],self.Rectangle2[2],self.Rectangle2[3])
        d = dDiff(d1,d2)
        return dDiff(d,d3)
    def BC(self,Node,Element):
        eps = 0.1 * np.sqrt((self.BdBox[1] - self.BdBox[0]) * (self.BdBox[3] - self.BdBox[2]) / np.size(Node, 0))
        FixedNode1 = np.sqrt((Node[:,0]-(self.BdBox[0]+0.100))**2+(Node[:,1]-self.BdBox[2])**2)
        FixedNode1 = np.argsort(FixedNode1)
        FixedNode2 = np.sqrt((Node[:,0]-(self.BdBox[1]-0.100))**2+(Node[:,1]-self.BdBox[2])**2)
        FixedNode2 = np.argsort(FixedNode2)
        LoadNode = np.sqrt((Node[:,0]-(self.Rectangle1[0]+self.Rectangle1[1])/2)**2+(Node[:,1]-self.BdBox[3])**2)
        LoadNode = LoadNode.argsort()
        FixedNode = np.array([FixedNode1[0],FixedNode2[0]])
        Supp = np.ones((len(FixedNode),3))
        Supp[:,0] = FixedNode
        Supp[1,1] = 0
        Load = np.array([[LoadNode[0],0,-1]])
        return Supp,Load
    def PFix(self):
        return np.array([[(self.BdBox[0]+0.100),self.BdBox[2]],
                         [(self.BdBox[1]-0.100),self.BdBox[2]],
                         [((self.Rectangle1[0]+self.Rectangle1[1])/2),self.BdBox[3]]])

class Lshape2Domain:
    def __init__(self):
        self.BdBox = np.array([0.0,2.0,0.0,1.0])
        self.Rect1 = np.array([0.0,0.4,0.0,0.4])
        self.Rect2 = np.array([1.2,1.6,0.4,0.6])
    def Dist(self,P):
        d1 = dRectangle(P, self.BdBox[0], self.BdBox[1], self.BdBox[2], self.BdBox[3])
        d2 = dRectangle(P, self.Rect1[0], self.Rect1[1], self.Rect1[2], self.Rect1[3])
        d3 = dRectangle(P, self.Rect2[0], self.Rect2[1], self.Rect2[2], self.Rect2[3])
        d = dDiff(d1, d2)
        return dDiff(d, d3)
    def BC(self,Node,Element):
        eps = 0.1 * np.sqrt((self.BdBox[1] - self.BdBox[0]) * (self.BdBox[3] - self.BdBox[2]) / np.size(Node, 0))
        FixedNode1 = np.sqrt((Node[:, 0] - (self.BdBox[0] + 0.200)) ** 2 + (Node[:, 1] - self.Rect1[3]) ** 2)
        FixedNode1=np.argsort(FixedNode1)
        FixedNode2 = np.sqrt((Node[:, 0] - (self.BdBox[1] - 0.200)) ** 2 + (Node[:, 1] - self.BdBox[2]) ** 2)
        FixedNode2=np.argsort(FixedNode2)
        LoadNode = np.sqrt(
            (Node[:, 0] - (self.BdBox[0] + self.BdBox[1]) / 2) ** 2 + (Node[:, 1] - self.BdBox[3]) ** 2)
        LoadNode = LoadNode.argsort()
        FixedNode = np.array([FixedNode1[0], FixedNode2[0]])
        Supp = np.ones((len(FixedNode), 3))
        Supp[:, 0] = FixedNode
        Supp[1, 1] = 0
        Load = np.array([[LoadNode[0], 0, -1]])
        return Supp, Load

    def PFix(self):
        return []


class Lshape3Domain:
    def __init__(self):
        self.BdBox = np.array([0,1.65,0,0.6])
        self.Rect1 = np.array([0,1.35,0,0.15])
        self.Rect2 = np.array([0.9,1.05,0.3,0.45])


    def Dist(self, P):
        d1 = dRectangle(P, self.BdBox[0], self.BdBox[1], self.BdBox[2], self.BdBox[3])
        d2 = dRectangle(P, self.Rect1[0], self.Rect1[1], self.Rect1[2], self.Rect1[3])
        d3 = dRectangle(P, self.Rect2[0], self.Rect2[1], self.Rect2[2], self.Rect2[3])
        d = dDiff(d1, d2)
        return dDiff(d, d3)

    def BC(self, Node, Element):
        eps = 0.1 * np.sqrt((self.BdBox[1] - self.BdBox[0]) * (self.BdBox[3] - self.BdBox[2]) / np.size(Node, 0))
        FixedNode1 = np.sqrt((Node[:, 0] - (self.BdBox[0] + 0.15)) ** 2 + (Node[:, 1] - self.Rect1[3]) ** 2)
        FixedNode1=np.argsort(FixedNode1)
        BottomNodes, = np.nonzero(abs(Node[:, 1] - self.BdBox[2]) < eps)
        LoadNode = np.sqrt( (Node[:, 0] - 0.6) ** 2 + (Node[:, 1] - self.BdBox[3]) ** 2)
        LoadNode = LoadNode.argsort()
        FixedNode = np.append(FixedNode1[0], BottomNodes)
        Supp = np.ones((len(FixedNode), 3))
        Supp[:, 0] = FixedNode
        Supp[0, 1] = 0
        Load = np.array([[LoadNode[0], 0, -1]])
        return Supp, Load

    def PFix(self):
        return []