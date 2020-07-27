import numpy as np
from grandpy.generateGsC import generateGS
from grandpy.GRANDpy import Grand, PlotGroundStructure
from grandpy.restrictionDomains import *
from grandpy.printBars import printBars
import matplotlib.pyplot as plt
import pickle

FileName='Lshaped3Domain11x4'
pickle_in = open('savedMeshersPolyMesher/LShape3Domain11x4.pickle','rb')
malha = pickle.load(pickle_in)
pickle_in.close()
Bars = generateGS(malha,6,restrictionLShape3)
#printBars(Bars,malha.Node)
#plt.show()
A = Grand(malha,Bars)

#SaveResult
pickle_out = open(f'resultsGrand/{FileName}.pickle','wb')
res = {'Mesher':malha,'Bars':Bars,'A':A}
pickle.dump(res,pickle_out)
pickle_out.close()
#FiguresSave
#SVG
Svgfil = open(f'resultsGrand/{FileName}.svg','wb')
plt.savefig(Svgfil,format='svg')
Svgfil.close()

#png
PngFil = open(f'resultsGrand/{FileName}.png','wb')
plt.savefig(PngFil,format='png')
PngFil.close()

plt.show()