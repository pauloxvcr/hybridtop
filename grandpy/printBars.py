import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection

def printBars(Bars,Node):
    fig,ax = plt.subplots()
    for bar in Bars:
        line = Line2D([Node[bar[0],0],Node[bar[1],0]],[Node[bar[0],1],Node[bar[1],1]],color='black',linewidth=0.2)
        ax.add_line(line)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.axis('off')
