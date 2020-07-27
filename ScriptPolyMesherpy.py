import pickle
from polymesherpy.polymesherpy import PolyMesher
from polymesherpy.polymesherpy import PolyMshr_PlotMsh
from polymesherpy.domains import *
import matplotlib.pyplot as plt

def Regular(domain,Nx,Ny,Lx,Ly):
    h = Lx / (Nx * 2)
    h2 = Ly / (Ny * 2)
    # Criar semente inicial domínio
    x, y = np.meshgrid(np.linspace(h, Lx - h, Nx), np.linspace(h2, Ly - h2, Ny))
    P = np.column_stack((x.flatten(), y.flatten()))
    d = domain.Dist(P)
    d = d[:, -1]
    P = P[d < 0]
    return P

FileName='ShearWall9x9'
domain = ShearWallFibDomain()

P = Regular(domain,9,9,8.1,8.1)



malha = PolyMesher(domain,2000,0,P)


#----------------------Saves
#PolyMesherResult
pickle_out = open(f'savedMeshersPolyMesher/{FileName}.pickle','wb')
pickle.dump(malha,pickle_out)
pickle_out.close()
#FiguresSave
#SVG
Svgfil = open(f'savedMeshersPolyMesher/{FileName}.svg','wb')
plt.savefig(Svgfil,format='svg')
Svgfil.close()

#png
PngFil = open(f'savedMeshersPolyMesher/{FileName}.png','wb')
plt.savefig(PngFil,format='png')
PngFil.close()

plt.show()