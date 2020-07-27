import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import identity
from scipy.sparse import lil_matrix
from numpy import ix_
from scipy.sparse import diags
from scipy.sparse import tril

from .restrictionDomains import NoneRestriction

from .compileversion import selectRemoveFlag

def generateGS(Mesher,Lvl,RestrictDomain=None,ColTol = 0.999999):
    Node,Elem,Supp,Load = Mesher.get()
    if RestrictDomain==None:
        RestrictDomain = NoneRestriction
    #Get element connectivity matrix
    Nn = max([max(Node) for Node in Elem])+1 # encontra o maior no para achar o numero de nós
    Ne = len(Elem) # quantos elementos tem
    A1 = lil_matrix((Nn,Nn)) # matriz sparsa tamanho Nn
    for i in range(0,Ne):
        A1[ix_(Elem[i],Elem[i])] = 1 #primeira situação é conecções no elemento
    A1 = A1 - identity(Nn) # retira conecção consigo msm
    An = A1 #
    #Level 1 connectivity
    I,J = An.nonzero()# onde tem conecção / ta o contrário pq matlab é por coluna ( acho que da pra ignorar aqui)
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
        I = np.delete(I,np.nonzero(RemoveFlag)[0])
        J = np.delete(J,np.nonzero(RemoveFlag)[0])

        newD = np.column_stack([Node[I,0]-Node[J,0],Node[I,1]-Node[J,1]])
        L = np.sqrt(newD[:,0]**2 +newD[:,1]**2).flatten()
        newD = np.column_stack([newD[:,0].flatten()/L,newD[:,1].flatten()/L]) # verificar necessidade de flatten() # vetor unitário direcional
        # Collinearity Check
        p=0 # de onde saem as barras
        m=0 # na teoria era pra ser quantidade de novas barras
        RemoveFlag=np.zeros(np.size(I)) # pode haver um non zero dps então vai ter que compensar !!!!!!!!!!!!!!!!!1
        Nb = np.size(Bars, 0)
        RemoveFlag = selectRemoveFlag(RemoveFlag,I,Bars,Nn,Nb,ColTol,D,newD)
         #alteração devido ao fato de python começar com 0
        '''Remove collinear bars and make sym[D[:,0].flatten()/L,D[:,1].flatten()/L]metric again. Bars that have one
        angle marked as collinear but the other not, will be spared
        '''
        ind, = np.nonzero(RemoveFlag==0)
        H = csr_matrix((np.ones(np.size(ind)),(I[ind],J[ind])),shape=(Nn,Nn))
        I,J = np.nonzero(H+H.T) #  garante a simetria e elimina a situação de o no ser eliminado em q e nao em p
        print(f'Lvl {i} - Collinear bars removed: {(len(RemoveFlag)-len(I))/2}')
        Bars = np.concatenate((Bars,np.column_stack([I,J])), axis=0)
        Bars = Bars[Bars[:,0].argsort()] # adiciona efetivamente as novas barras
        D = np.column_stack([Node[Bars[:,0],0]-Node[Bars[:,1],0],Node[Bars[:,0],1]-Node[Bars[:,1],1]]) #fazer o vetor unitario direcional
        L = np.sqrt(D[:,0]**2 +D[:,1]**2)#
        D = np.column_stack([D[:,0].flatten()/L,D[:,1].flatten()/L])
    A= csr_matrix((np.ones(np.size(Bars,0)),(Bars[:,0],Bars[:,1])),shape=(Nn,Nn)) # encerra, mas ainda precisa retirar barras repetidas
    I,J = tril(A).nonzero() # para isso usa somente o triangulo superior
    Bars = np.column_stack([I,J])
    return Bars