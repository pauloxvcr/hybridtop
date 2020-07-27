from polymesherpy.polymesherpy import PolyMesher
import polymesherpy.domains

from polytoppy.polytoppy import PolyTop
from polytoppy.polytoppy import FiniteElement
from polytoppy.polyFilter import PolyFilter
from polytoppy.matIntFnc import simp

import matplotlib.pyplot as plt

#usando pickle para acelerar
import pickle
#Observação O Runtime Error é devido a plotBoundary pq tem matrix com Nan e há comparação >
FileName = 'Hpier 0.3vol'
#open mesher
pickle_in = open('savedMeshersPolyMesher/HPier2000.pickle', 'rb')
malha = pickle.load(pickle_in)
pickle_in.close()

fem = FiniteElement(malha)
P = PolyFilter(fem)
m = simp
opt = PolyTop(0.3,P,m)
opt.set_fem(fem)
penal = opt.penal
while penal <= 4:
    opt.penal = penal
    opt.Opt()
    penal = penal+0.5
    print(f'penal : {penal}')
opt.polyPlot()

#Save Result
Result = opt.save_result()
File = open(f'resultsPolyTop/{FileName}.pickle','wb')
pickle.dump(Result,File)
File.close()
#FiguresSave
#SVG
Svgfil = open(f'resultsPolyTop/{FileName}.svg','wb')
plt.savefig(Svgfil,format='svg')
Svgfil.close()

#png
PngFil = open(f'resultsPolyTop/{FileName}.png','wb')
plt.savefig(PngFil,format='png')
PngFil.close()


plt.show()