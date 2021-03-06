# Hybridtoppy

This repository contain a collection of algorithms that is useful for topology optimization.
The collection is composed by 3 Algorithms translated from Matlab to python [1-3] 
and a new algorithm that accomplish hybrid topology optimization based on the formulation proposed by Gaynor et al[4].

The 3 Softwares, PolyMesherpy, GRANDpy and PolyToppy, are for educational or academic use only. All rights of reproduction or
distribution in any form are reserved to the original creators.

The Matlab algorithms and the algorithms articles can be found in:

https://paulino.ce.gatech.edu/software.html

The contact to the creators of the 3 MatLab algorithms can be found in:

http://paulino.ce.gatech.edu/contact.html

The HybridTop algorithm was part of a master degree thesis in civil engineering  oriented by Rejane Canha.

## Algorithms
### PolyMesher[1]
 "Is a simple and robust MATLAB *(here converted to python)* code for polygonal mesh generation, called PolyMesher."(https://paulino.ce.gatech.edu/software.html)
### GRAND[2]
"GRound structure Analysis and Design is an educational MATLAB *(here converted to python)* code for structural topology optimization with discrete elements using the ground structure approach."    (https://paulino.ce.gatech.edu/software.html)
### PolyTop[3]
"PolyTop is an efficient Matlab *(here converted to python)* code for structural topology optimization that includes a general finite element routine based on isoparametric polygonal elements."    (https://paulino.ce.gatech.edu/software.html)
### HybridToppy
HybridTopy is a python code for structural topology optimization focused on Reinforced Concrete. This algorithm considers concrete as a continuum (using isoparametric polygonal elements) and the reinforcement as bar elements(using unidimensional finite element).

## How to use
The repository has Scripts for all algorithms that is organized for direct use.

### 1 - Mesher Generation
  If there is not a mesh, the mesh can be obtain with the use of the ScriptPolyMesherpy.  
The PolyMesherpy will generate a poligonal mesh using sign and dist functions to describe the domain. Details of how this work are in [1]

### 2 - Chose the topology structural optimization type
(a) Truss Optimization -> GRANDpy  
(b) Continuum Optimization -> PolyToppy  
\(c\) Hybrid Optimization (for reinforced concrete) -> HybridToppy  

### 2.c - If the option is c you will need a hybrid mesh.
The hybrid mesh can be generated with the use of the ScripHybridMesher.  
For this, you will need two meshes, thin and sparse.

### 3 - Obtain the result
Use the following Scripts according to the chosen optimization method:  
(a) ScriptGrandpy    
(b) ScriptPolyToppy    
\(c\) ScriptHybridToppy  

---

## Package requirements:
- numpy,scipy -> https://numpy.org/
 - matplotlib -> https://matplotlib.org/  
This packages are include in conda  

If use generategsc on grand or HybridToppy with multicore (default configuration):
- numba -> http://numba.pydata.org/

---

## References:
### MatLab Algorithms:
---
#### PolyMesher  

[1] Talischi C, Paulino GH, Pereira A, Menezes IFM (2012) PolyMesher: a general-purpose mesh generator for polygonal elements written in Matlab. Structural and Multidisciplinary Optimization 45:309–328. https://doi.org/10.1007/s00158-011-0706-z  

---
  #### GRAND
  [2]Zegard T, Paulino GH (2014) GRAND — Ground structure based topology optimization for arbitrary 2D domains using MATLAB. Structural and Multidisciplinary Optimization 50:861–882. https://doi.org/10.1007/s00158-014-1085-z

---
  #### PolyTop
[3] Talischi C, Paulino GH, Pereira A, Menezes IFM (2012) PolyTop: a Matlab implementation of a general topology optimization framework using unstructured polygonal finite element meshes. Structural and Multidisciplinary Optimization 45:329–357. https://doi.org/10.1007/s00158-011-0696-x

---

### Hybrid Formulation:
[4]Gaynor AT, Guest JK, Moen CD (2013) Reinforced Concrete Force Visualization and Design Using Bilinear Truss-Continuum Topology Optimization.Journal of Structural Engineering 139:607–618. https://doi.org/10.1061/(ASCE)ST.1943-541X.0000692


---

## Observations


Author of the translation and HybridToppy code: 
Paulo Vinicius Costa Rodrigues 

I thank UFS — *Universidade Federal de Sergipe* —, Proec — *Programa de Pós-Graduação em Engenharia Civil* — and Capes — *Coordenação de Aperfeiçoamento de Pessoal de Nível Superior* — for their support.

---

For now the algorithms do not follow the python convention, this made the translation easier.

Some things I change the logic because a "literal" translation would not work.

---
contact: pauloxvcr@gmail.com


