import numpy as np

def rCircle(C,r,Node,Bars):
    Nb = np.size(Bars,0)
    C = C.reshape((1, 2))
    U = Node[Bars[:,0],:] - np.repeat(C,Nb,0)
    V = Node[Bars[:,1],:] - np.repeat(C,Nb,0) # vector center final node
    D = V - U # vector of the bar
    L = np.sqrt(D[:,0]**2 + D[:,1]**2) # Bar L
    D = np.column_stack((D[:,0]/L,D[:,1]/L)) #Normatize the bar direction
    flag = np.logical_or((np.sum(D*V,1)>=0) * (np.sum(D*U,1)<=0) * ( abs(D[:,0] *U[:,1]-D[:,1] *U[:,0])<r ), # Condição em que primeiro se ver a direção do D e depois verifica se passa secantemente pelo circulo através de um produto vetorial( o seno vai fazer a parte da distância da reta ao centro pois D é unitário)
                         ( U[:,0]**2+U[:,1]**2<=r**2 ))
    flag = np.logical_or(flag,( V[:,0]**2+V[:,1]**2<=r**2 ))
    return flag
def rRectangle(Amin,Amax,Node,Bars):
    #Amin and Amax are the rectangle's limit coords: minimum and maximum
    Nb = np.size(Bars,0)
    Tmin = np.zeros(Nb)
    Tmax = np.ones(Nb)
    D = Node[Bars[:,1],:] - Node[Bars[:,0],:]
    for i in range(0,2): # verifica por slab(entreplanos) se a entrada mais longe no entreplanos tiver mais longe que a saida mais proxima não há intersecção
        T1 = (Amin[i]-Node[Bars[:,0],i]) / D[:,i]
        T2 = (Amax[i]-Node[Bars[:,0],i]) / D[:,i]
        ind = np.nonzero(T1>T2)
        T1[ind],T2[ind] = T2[ind],T1[ind]
        Tmin = np.maximum(Tmin,T1)
        Tmax = np.minimum(Tmax,T2)
    flag = (Tmin<=Tmax)
    return flag

def rLine(A,B,Node,Bars):
    # Line segment between points A and B
    P = Node[Bars[:, 0], :]
    D = Node[Bars[:, 1], :] - P
    V = B - A
    C = D[:, 0] * V[1] - V[0] * D[:, 1]
    Ct = (A[0] - P[:, 0]) * D[:, 1] - (A[1] - P[:, 1]) * D[:, 0]
    Cu = (A[0] - P[:, 0]) * V[1] - (A[1] - P[:, 1]) * V[0]
    Ct = Ct / C
    Cu = Cu / C
    flag = (Ct>0)*(Ct<1)*(Cu>0)*(Cu<1)
    return flag

def rLine2(A,B,Node,Bars):
    P = Node[Bars[:, 0], :]

    D = Node[Bars[:, 1], :] - P


    V = B - A


    det = -V[0]*D[:,1] +D[:,0]*V[1]
    t1 = 1/det * (D[:,1]*(A[0]-P[:,0])-D[:,0]*(A[1]-P[:,1]))
    t2 = 1/det * (V[1]*(A[0]-P[:,0])-V[0]*(A[1]-P[:,1]))

    flag = (t1>0)*(t1<1)*(t2>0)*(t2<1)
    return flag

