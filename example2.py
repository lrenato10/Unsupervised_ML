#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:20:33 2023

@author: lurodrig
"""

import numpy as np
import matplotlib . pyplot as plt
import time
#from sklearn import cluster
from sklearn import * #cluster, datasets, svm, metrics
print(dir(datasets))

from scipy . io import arff
from sklearn.metrics import silhouette_samples, silhouette_score

#%% Exercice 1
#Import dataset
path = '/home/lurodrig/Unsupervised_ML/artificial/'
data_set = ["2d-4c-no9.arff", "smile1.arff", "spiral.arff"]
databrut = arff . loadarff ( open ( path + data_set[2] , 'r') )
datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in databrut [ 0 ] ]
datanp_array=np.array(datanp)
#Extract Columns 0 and 1 
f0 = datanp_array[:, 0] # tous les elements de la premiere colonne
f1 = datanp_array[: ,1] # tous les elements de la deuxieme colonne
plt.scatter(f0, f1, s = 8)
plt.title("Donnees initiales")
plt.show()


#%% Exercice 2-1
#
# Les donnees sont dans datanp ( 2 dimensions )
# f0 : valeurs sur la premiere dimension
# f1 : valeur sur la deuxieme dimension
#
print ( "Appel KMeans pour une valeur fixee de k" )
tps1 = time.time()
k = 3
model = cluster.KMeans(n_clusters =k , init = 'k-means++' )
model.fit( datanp_array )

tps2 = time.time()
labels = model.labels_
iteration = model.n_iter_
plt.scatter ( f0 , f1 , c = labels , s = 8 )
plt.title ( " Donnees apres clustering Kmeans " )
plt.show()
print (" nb clusters = " ,k , " , nb iter = " , iteration , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )

#%% Exercice 2-2
#sklearn.metrics.silhouette_score(X, labels, *, metric='euclidean', sample_size=None, random_state=None, **kwds)
sklearn.metrics.silhouette_score(datanp_array, labels, metric='euclidean', sample_size=None, random_state=None)