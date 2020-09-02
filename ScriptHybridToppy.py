import numpy as np
import matplotlib.pyplot as plt

from polymesherpy.polymesherpy import PolyMesher,PolyMshr_PlotMsh
from polymesherpy.domains import MbbDomain
from grandpy.printBars import printBars
from hybridtoppy.finiteElement import FiniteElementStructure
from hybridtoppy.hybridtoppy import HybridTop
from hybridtoppy.polyFilter import PolyFilter
from hybridtoppy import matIntFnc,matIntFncBar
#usando pickle para acelerar
import pickle
from grandpy.printBars import printBars

import winsound
#
FileName = 'Lshape3'
pickle_in = open('savedMeshersHybridMesher/Lshape3.pickle', 'rb')
vol = 0.3
malha = pickle.load(pickle_in)
pickle_in.close()
'''printBars(malha[4], malha[0])
plt.show()'''

fem = FiniteElementStructure(malha)
P = PolyFilter(fem)
m = matIntFnc.simp
m2 = matIntFncBar.simp

opt = HybridTop(vol,P,m,m2)
opt.set_fem(fem)

penal = opt.penal
while penal <= 4:
    print(f'penal : {penal}')
    opt.penal = penal
    opt.Opt()
    penal = penal+0.5
opt.polyPlot(tol=0.0001)


#Save Result
Result = opt.save_result()
File = open(f'resultsHybridTop/{FileName}.pickle','wb')
pickle.dump(Result,File)
File.close()
#FiguresSave
#SVG
Svgfil = open(f'resultsHybridTop/{FileName}.svg','wb')
plt.savefig(Svgfil,format='svg')
Svgfil.close()

#png
PngFil = open(f'resultsHybridTop/{FileName}.png','wb')
plt.savefig(PngFil,format='png')
PngFil.close()

winsound.Beep(1000,1000)
plt.show()



