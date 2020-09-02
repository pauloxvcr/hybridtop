import numpy as np
from numba import njit
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

#-------------compiled for two-dimensional----------------------------
@njit(nogil = True)
def LocalKbi(eNode,Node, W,SdNdxi, Nu0,Ec,Et, sigmap, theta):
    E1 = E_concrete(sigmap[0],Ec,Et)
    E2 = E_concrete(sigmap[1],Ec,Et)
    E12 = (E1*E2)**0.5
    veff = (E12/Ec)*Nu0
    Dp = 1 / (1 - veff ** 2) * np.array([[E1, veff*E12, 0], [veff*E12, E2, 0], [0, 0, 1/4*(E1+E2-2*veff*E12)]])
    Qe = np.array([[(np.cos(theta)) ** 2, (np.sin(theta)) ** 2, np.sin(theta) * np.cos(theta)],
                   [(np.sin(theta)) ** 2, (np.cos(theta)) ** 2, -np.sin(theta) * np.cos(theta)],
                   [-2 * np.sin(theta) * np.cos(theta), 2 * np.sin(theta) * np.cos(theta), (np.cos(theta)) ** 2 - (np.sin(theta)) ** 2]])
    D = Qe.T @ Dp @ Qe
    nn = eNode.size
    Ke = np.zeros((2 * nn, 2 * nn))
    DB = np.zeros((3, 2 * nn))
    for q in range(0, len(W)):
        dNdxi = SdNdxi[:, :, q]
        J0 = Node[eNode, :].T @ dNdxi
        dNdx = dNdxi @ np.linalg.inv(J0)  # check better solution for this
        B = np.zeros((3, 2 * nn))
        B[0, 0: 2 * nn:2] = dNdx[:, 0]
        B[1, 1: 2 * nn:2] = dNdx[:, 1]
        B[2, 0: 2 * nn: 2] = dNdx[:, 1]
        B[2, 1: 2 * nn: 2] = dNdx[:, 0]
        temp = D @ B * W[q] * np.linalg.det(J0)
        Ke = Ke + B.transpose() @ temp
        DB = DB + temp  # For speed
    return Ke, DB
#Material Bilinear formate
@njit(nogil=True)
def E_concrete(sigmap,Ec,Et):
    if sigmap<=0: return Ec
    else: return Et

@njit(nogil = True)
def calcsigma(DB,Ue,ElemArea):
    return (DB @ Ue)/ElemArea
@njit(nogil = True)
def calctheta(sigma):
    theta = 1/2*np.arctan2(2*sigma[2],(sigma[0]-sigma[1])) # quando é x,y são 0 arctan2 é zero
    t = np.array([[np.cos(theta),-np.sin(theta)],
                 [np.sin(theta),np.cos(theta)]])
    sp = t.T @ np.array([[sigma[0],sigma[2]],[sigma[2],sigma[1]]]) @ t
    sigmap = np.array([sp[0,0],sp[1,1],sp[0,1]])
    return sigmap, theta
# ---------------- compiled for bars ----------------
@njit(nogil = True)
def calcdeformation(T,l,Ue):
    Ul = T @ Ue
    e = (Ul[1]-Ul[0])/l
    if e[0] <=0: # numba modification compatibility
        BarType = 0
    else:
        BarType = 1
    return BarType

# ------ Para uso em multtherading ----------------------
#------------------------------------------------------

#--------------------- Bidimensional -------------------
def bidimensionalElementUpdate(bidimensionalelement,U, Node, ShapeFnc, Nu0,Ec,Et ):
    bidimensionalelement.setUe(U)
    bidimensionalelement.LocalK(Node, ShapeFnc, Nu0,Ec,Et)
    bidimensionalelement.calcsigma()
    bidimensionalelement.calctheta()
def bidimensionalUpdate(bidimensionalelement,NElem,U,Node, ShapeFnc, Nu0,Ec,Et ):
    def bfunction(i):
        bidimensionalElementUpdate(bidimensionalelement[i],U,Node, ShapeFnc, Nu0, Ec,Et)

    '''for i in range(NElem):
        bfunction(i)'''
    with ThreadPoolExecutor(multiprocessing.cpu_count()) as ex:
        ex.map(bfunction,range(NElem))

#------------------- Bars ------------------------------
def barElementUpdate(bar, U):
    bar.setUe(U)
    bar.calcdeformation()
def barsUpdate(Bars,NBars,U):
    def bfunction(i):
        barElementUpdate(Bars[i],U)
    with ThreadPoolExecutor(4) as ex:
        ex.map(bfunction,range(NBars))
    '''for i in range(NBars):
        bfunction(i)'''
