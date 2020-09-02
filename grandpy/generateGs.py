import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import identity
from scipy.sparse import lil_matrix
from numpy import ix_
from scipy.sparse import diags
from scipy.sparse import tril

from .restrictionDomains import NoneRestriction

def generateGS(Node,Elem,Lvl,RestrictDomain=None,ColTol = 0.999999):
    if RestrictDomain==None:
        RestrictDomain = NoneRestriction
    #Get element connectivity matrix
    Nn = max([max(Node) for Node in Elem])+1 # find the largest node to find the number of nodes
    Ne = len(Elem) # how many elements are there
    A1 = lil_matrix((Nn,Nn)) # sparse matrix
    for i in range(0,Ne):
        A1[ix_(Elem[i],Elem[i])] = 1 #first situation is connections in the element
    A1 = A1 - identity(Nn) # disconnect from yourself
    An = A1 #
    #Level 1 connectivity
    I,J = An.nonzero()# where there is a connection / this is the opposite because matlab is per column
    Bars = np.column_stack([I,J])
    D = np.column_stack([Node[I,0] - Node[J,0],Node[I,1] - Node[J,1]])
    L = (np.sqrt(D[:,0]**2 + D[:,1]**2))
    D = np.column_stack([D[:,0].flatten()/L,D[:,1].flatten()/L])
    #Levels 2 and above
    for i in range(1,Lvl):
        Aold = An
        An = (An*A1).astype(bool)
        Gn = An - Aold
        Gn.setdiag(0)
        I,J = np.nonzero(Gn)
        if len(J) == 0:
            Lvl = i -1
            print(f'-INFO- No new bars at Level {Lvl}')
            break
        RemoveFlag = RestrictDomain(Node,np.column_stack([I,J])) #
        I = np.delete(I,RemoveFlag)
        J = np.delete(J,RemoveFlag)

        newD = np.column_stack([Node[I,0]-Node[J,0],Node[I,1]-Node[J,1]])
        L = np.sqrt(newD[:,0]**2 +newD[:,1]**2).flatten()
        newD = np.column_stack([newD[:,0].flatten()/L,newD[:,1].flatten()/L])
        # Collinearity Check
        p=0 #
        m=0 #
        RemoveFlag=np.zeros(np.size(I),dtype=bool) #
        Nb = np.size(Bars,0)
        for j in range(0,Nn):
            #Find I(p:q) - NEW bars starting @ node 'j' ( I está em ordem então p e q servem para achar as barras que saem do no j=p sendo q-1 o ultimo indice
            for p in range(p,len(I)):
                if I[p]>=j:
                    break
            for q in range(p,len(I)):
                if I[q]>j:
                    break
            if  I[q]> j: #
                q = q-1

            if I[p] == j: #
                #Find BARS(m:n) - OLD bars starting @ node 'j'
                for m in range(0,Nb):
                    if Bars[m,0]>=j: #
                        break
                for n in range(m,Nb):
                    if Bars[n,0]>j:
                        break
                if Bars[n,0]>j:
                    n = n-1
                if Bars[n,0] == j:
                    # Dot products of old vs. new bars. If ~collinear: mark
                    C = np.max(D[m:n+1,:] @ newD[p:q+1,:].T,axis=0) #
                    RemoveFlag[p+np.argwhere(C>ColTol)] = True #change due to the fact that python starts with 0
            '''Remove collinear bars and make sym[D[:,0].flatten()/L,D[:,1].flatten()/L]metric again. Bars that have one
            angle marked as collinear but the other not, will be spared
            '''
        ind, = np.nonzero(RemoveFlag==0)
        H = csr_matrix((np.ones(np.size(ind)),(I[ind],J[ind])),shape=(Nn,Nn))
        I,J = np.nonzero(H+H.T) #  guarantees symmetry and eliminates the situation of the node being eliminated in q and not in p
        print(f'Lvl {i} - Collinear bars removed: {(len(RemoveFlag)-len(I))/2}')
        Bars = np.concatenate((Bars,np.column_stack([I,J])), axis=0)
        Bars = Bars[Bars[:,0].argsort()] # effectively adds the new bars
        D = np.column_stack([Node[Bars[:,0],0]-Node[Bars[:,1],0],Node[Bars[:,0],1]-Node[Bars[:,1],1]]) #directional unit vector
        L = np.sqrt(D[:,0]**2 +D[:,1]**2)#
        D = np.column_stack([D[:,0].flatten()/L,D[:,1].flatten()/L])
    A= csr_matrix((np.ones(np.size(Bars,0)),(Bars[:,0],Bars[:,1])),shape=(Nn,Nn)) # ends, but still needs to remove repeated bars
    I,J = tril(A).nonzero() # for this use only the upper triangle
    Bars = np.column_stack([I,J])
    return Bars