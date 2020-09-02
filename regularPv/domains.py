"Broken, will be removed"
import numpy as np
class RectangleFormat:
    def __init__(self,x1,y1,x3,y3):
        self.x1 = x1
        self.x3 = x3
        self.y1 = y1
        self.y3 = y3
        self.type = 0
    def is_inside(self,Nodes):
        inside = np.logical_and(np.logical_and((Nodes[:,0]>self.x1) , (Nodes[:,0]<self.x3)),
                 np.logical_and((Nodes[:,1]>self.y1),(Nodes[:,0]<self.y3)))
        return inside




class GenericDomain():
    def __init__(self,BdBox):
        self.BdBox = BdBox
        self.ishapes = []
        self.rshapes = []
    def build(self,Nxy):
        Nodes, Elem = self.initialGrid(Nxy)
        iNodes = self.verify(self.ishapes,Nodes)
        iNodes = np.logical_and(iNodes,(not self.verify(self.rshapes,Nodes)))
        rNodes = np.nonzero(not iNodes)
        for i in range(len(Elem)):
            for node in Elem[i]:
                if node in rNodes:
                    del Elem[i]
        self.Nodes = Nodes[iNodes]
        self.Elem = Elem

    def insertRect(self,rect):
        self.ishapes.append(rect)
    def removeRect(self,rect):
        self.rshapes.append(rect)

    def initialGrid(self,Nxy):
        Nx = Nxy[0]
        Ny = Nxy[1]
        x, y = np.meshgrid(np.linspace(self.BdBox[0], self.BdBox[2], Nx + 1),
                           np.linspace(self.BdBox[1], self.BdBox[3], Ny + 1))
        Nodes = np.column_stack((x.flatten(), y.flatten()))
        Elem = []
        for i in range(0, Ny):
            for j in range(0, Nx):
                n1 = i * (Nx + 1) + j
                n2 = (i + 1) * (Nx + 1) + j
                Elem.append([n1, n1 + 1, n2 + 1, n2])
        return Nodes,Elem

    def rect_to_trap(self, rect, trapezeNodes):
        rectNodesi = np.nonzero(rect.is_inside(self.Nodes))
        rectNodes = self.Nodes[rectNodesi,:]

        X1 = rect.x1
        Y1 = rect.Y1
        X3 = rect.x3
        Y3 = rect.y3
        x1,y1,x2,y2,x3,y3,x4,y4 = trapezeNodes

        xt = np.array([x1,x2,x3,x4])
        yt = np.array([y1,y2,y3,y4])
        sysMatrix = np.array([[X1,Y1,X1*Y1,1],
                             [X3,Y1,X3*Y1,1],
                             [X3,Y3,X3*Y3,1],
                             [X1,Y3,X1*Y3,1]])

        c1 = np.linalg.solve(sysMatrix,xt)
        c2 = np.linalg.solve(sysMatrix,yt)

        X = rectNodes[:,0]
        Y = rectNodes[:,1]

        x = c1[0]*X+c1[1]*Y+c1[2]*X*Y+c1[3]
        y = c2[0]*X+c2[1]*Y+c2[2]*X*Y+c2[3]
        rectNodes = np.column_stack([x,y])
        self.Nodes[rectNodesi,:] = rectNodes
        return

    def verify(self,shapes,Nodes):
        inside = np.zeros(np.size(Nodes),dtype=bool)
        for shape in shapes:
            inside = np.logical_or(inside,shape.verify(Nodes))
        return inside


    def BC(self,Node,Element):
        pass

    def NodesFix(self):
        pass

