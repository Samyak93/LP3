import numpy as np
import pandas as pd 
import matplotlib.pyplot as mp
X=[[0.1,0.6],[0.15,0.71],[0.08,0.9],[0.16,0.15],[0.2,0.3],[0.25,0.5],[0.24,0.1],[0.3,0.2]]
centres = np.array([[0.1,0.6],[0.3,0.2]])
print ('Initial Centroids : \n',centres)
from sklearn.cluster import KMeans
model= KMeans(n_clusters=2,init=centres,n_init=1)
model.fit(X)
print('Labels:',model.labels_)
print('P6 belongs to cluster:',model.labels_[5])
print('number of population around cluster2:',np.count_nonzero(model.labels_==1))
print('New Centroids:',model.cluster_centers_)
