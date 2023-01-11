#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 15:59:16 2023

@author: lurodrig
"""

import numpy as np
import matplotlib . pyplot as plt
from scipy . io import arff
# Parser un fichier de donnees au format arff
# data est un tableau d ’ exemples avec pour chacun
# la liste des valeurs des features
#
# Dans les jeux de donnees consideres :
# il y a 2 features ( dimension 2 )
# Ex : [[ - 0 . 499261 , -0 . 0612356 ] ,
# [ - 1 . 51369 , 0 . 265446 ] ,
# [ - 1 . 60321 , 0 . 362039 ] , .....
# ]
#
# Note : chaque exemple du jeu de donnees contient aussi un
# numero de cluster . On retire cette information
path = '/home/lurodrig/Unsupervised_ML/artificial/'
data_set = "s-set2.arff"
databrut = arff . loadarff ( open ( path + data_set , 'r') )
datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in databrut [ 0 ] ]
# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ - 0 . 499261 , -1 . 51369 , -1 . 60321 , ...]
# Ex pour f1 = [ - 0 . 0612356 , 0 . 265446 , 0 . 362039 , ...]
datanp_array=np.array(datanp)

f0 = datanp_array[:, 0] # tous les elements de la premiere colonne
f1 = datanp_array[: ,1] # tous les elements de la deuxieme colonne
plt.scatter(f0, f1, s = 8)
plt.title("Donnees initiales")
plt.show()
