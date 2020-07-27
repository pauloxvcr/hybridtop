from .restrictions import *
import numpy as np

def NoneRestriction(*args):
    return []

def restrictionShearDomain(Node,Bars):
    tol = 1e-3
    flag1 = rRectangle((3.6+tol,0),(4.5-tol,1.8-tol),Node,Bars)
    flag2 = rRectangle((3.6+tol,2.7+tol),(4.5-tol,4.5-tol),Node,Bars)
    flag3 = rRectangle((3.6+tol,5.4+tol),(4.5-tol,7.2-tol),Node,Bars)
    flag = np.logical_or(flag3,np.logical_or(flag1,flag2))
    return flag

def restrictionHPierDomain(Node,Bars):
    eps = 0.001
    flag1 = rLine(np.array([1/12+eps,0]),np.array([1/12+eps,1/2-eps]),Node,Bars)
    flag2 = rLine(np.array([1/12+eps,1/2-eps]),np.array([1/2,23/36-eps]),Node,Bars)
    flag = np.logical_or(flag1,flag2)
    return flag

def restrictionMichell(Node,Bars):
    eps = 0.001
    flag = rCircle(np.array([0,0]),1-eps,Node,Bars)
    return flag

def restrictionLShape(Node,Bars):
    tol = 1e-3
    flag1 = rRectangle((0.3 + tol, 0.2 + tol), (0.7 - tol, 0.4 - tol), Node, Bars)
    flag2 = rRectangle((1 + tol, 0.3 + tol), (1.4 - tol, 0.6 - tol), Node, Bars)
    flag = np.logical_or(flag1,flag2)
    return flag
def restrictionLShape2(Node,Bars):
    tol = 1e-3
    flag1 = rRectangle((0.0 + tol, 0.0 + tol), (0.4 - tol, 0.4 - tol), Node, Bars)
    flag2 = rRectangle((1.2 + tol, 0.4 + tol), (1.6 - tol, 0.6 - tol), Node, Bars)
    flag = np.logical_or(flag1,flag2)
    return flag
def restrictionLShape3(Node,Bars):
    tol = 1e-3
    flag1 = rRectangle((0.0+tol,0+tol),(1.35-tol,0.15-tol),Node,Bars)
    flag2 = rRectangle((0.9+tol,0.3+tol),(1.05-tol,0.45-tol),Node,Bars)
    flag = np.logical_or(flag1,flag2)
    return flag