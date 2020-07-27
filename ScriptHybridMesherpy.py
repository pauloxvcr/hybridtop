import pickle

from hybridmesherpy.hybridmesher import HybridMesher


from grandpy.printBars import printBars
import matplotlib.pyplot as plt

from grandpy.restrictionDomains import *
FileName = 'Mbb_fpoly_epoly'
#Malha Fina
pickle_in = open('savedMeshersPolyMesher/Mbb/Mbb3000.pickle','rb')
malha1 = pickle.load(pickle_in)
pickle_in.close()
#Malha Grossa
pickle_in = open('savedMeshersPolyMesher/Mbb/Mbb32.pickle','rb')
malha2 = pickle.load(pickle_in)
pickle_in.close()
if len(malha1.Element)<len(malha2.Element):
    raise('Erro : malha 1 mais esparsa que malha 2')


malha = HybridMesher(malha1,malha2,6)




pickle_out = open(f'savedMeshersHybridMesher/{FileName}.pickle','wb')
pickle.dump(malha,pickle_out)
pickle_out.close()

printBars(malha.Bars,malha.Node)

#FiguresSave
#SVG
Svgfil = open(f'savedMeshersHybridMesher/{FileName}.svg','wb')
plt.savefig(Svgfil,format='svg')
Svgfil.close()

#png
PngFil = open(f'savedMeshersHybridMesher/{FileName}.png','wb')
plt.savefig(PngFil,format='png')
PngFil.close()

plt.show()


