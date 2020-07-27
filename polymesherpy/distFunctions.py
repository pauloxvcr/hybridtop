import numpy as np

def dRectangle(P,x1,x2,y1,y2):
    d = np.array([x1-P[:,0], P[:,0]-x2, y1-P[:,1], P[:,1]-y2]).T
    d = np.column_stack((d,np.max(d,1)))
    return d

def dCircle(P,xc,yc,r):
    d = np.sqrt((P[:, 0] - xc)** 2 + (P[:, 1] - yc)** 2)-r
    return np.array([d,d]).transpose() #array 2d

def dDiff(d1,d2):
    d = np.concatenate((d1[:,0:-1],d2[:,0:-1]), axis=1)
    d = np.insert(d,np.size(d,1),np.maximum(d1[:,-1],-d2[:,-1]),axis=1)
    return d

def dDIntesect(d1,d2):
    d = np.concatenate((d1[:,0:-1],d2[:,0:-1]), axis = 1)
    d = np.insert(d,np.size(d,1),np.maximum(d1[:,-1],d2[:,-1]), axis =1)
    return d
def dLine(P,x1,y1,x2,y2):
    a = np.array([x2 - x1, y2 - y1])
    a = a / np.linalg.norm(a)
    b = np.array([P[:, 0]-x1, P[:, 1]-y1]).T
    d = b[:, 0]*a[1] - b[:, 1]*a[0]
    d = np.column_stack((d,d))
    return d

def dUnion(d1,d2):
    d = np.concatenate((d1[:,0:-1],d2[:,0:-1]), axis =1 )
    d = np.insert(d,np.size(d,1),np.minimum(d1[:,-1],d2[:,-1]) ,axis =1)
    return d
