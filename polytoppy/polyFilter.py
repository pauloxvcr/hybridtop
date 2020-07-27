import numpy as np
from scipy.sparse import identity
from scipy.sparse import csc_matrix
from scipy.sparse import diags

def PolyFilter(fem,R=None,mult=4):
    if R==None:
        mArea = sum(abs(fem.ElemArea))/fem.NElem
        R = ((mArea/(np.pi))**0.5)*mult
    elif R<0: #P is set to identity when R<0
        P= identity(fem.NElem)
        return P
    ElemCtrd = np.zeros((fem.NElem,2))
    for el in range(0,fem.NElem): #%Compute the centroids of all the elements
        vx=fem.Node[fem.Element[el],0]
        vy = fem.Node[fem.Element[el],1]
        temp = vx * np.append(vy[1:len(vy)],vy[0]) - vy * np.append(vx[1:len(vy)],vx[0])
        A = 0.5 * sum(temp)
        ElemCtrd[el,0] = 1/(6*A)*sum((vx+np.append(vx[1:len(vy)],vx[0])) * temp)
        ElemCtrd[el,1] = 1/(6*A)*sum((vy+np.append(vy[1:len(vy)],vy[0])) * temp)
    d = DistPntSets(ElemCtrd,ElemCtrd,R) #%Obtain distance values & indices
    P = csc_matrix((1-d[:,2]/R, (d[:,0].astype(int), d[:,1].astype(int)))) #Assemble the filtering matrix obs: como em distpntSets somente  salva os valores de distância menores que R respeita a condição da função bola
    P = diags(1/ np.array(np.sum(P,1)).flatten(),0,shape=(fem.NElem,fem.NElem)) @ P # aplica o coeficiente de correção cp
    return P
#-------------------------------------COMPUTE DISTANCE BETWEEN TWO POINT SETS------------------------------------------
def DistPntSets(PS1,PS2,R):
    d = np.empty([0,3])
    for el in range(0,np.size(PS1,0)):
        dist = np.sqrt((PS1[el,0]-PS2[:,0])**2 + (PS1[el,1]-PS2[:,1])**2)
        I, = np.nonzero(dist<=R)
        J = np.zeros(np.size(I),dtype=int)
        temp = np.stack([I,J+el,dist[I]]).T
        d = np.append(d,temp,axis=0)
    return d
