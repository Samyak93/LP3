###########LR
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset  = pd.read_csv("LR.csv")
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)

accuracy = regressor.score(x,y)*100
print(accuracy)

y_pred = regressor.predict([[10]])
print(y_pred)

hours = int(input("Enter hours"))
eq = regressor.coef_*hours+regressor.intercept_
print(eq[0])

plt.plot(x,y,'o')
plt.plot(x,regressor.predict(x))
plt.show()





########KNN
import pandas as pd
import numpy as np

#Read dataset
dataset=pd.read_csv("KNN.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,2].values

#import KNeighborshood Classifier and create object of it
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=3)
classifier.fit(X,y)

#predict the class for the point(6,6)
X_test=np.array([6,2])
y_pred=classifier.predict([X_test])
print('General KNN',y_pred)

classifier=KNeighborsClassifier(n_neighbors=3,weights='distance')
classifier.fit(X,y)

#predict the class for the point(6,6)
X_test=np.array([6,2])
y_pred=classifier.predict([X_test])
print('Distance Weighted KNN',y_pred)





###################DC
import pandas as pd
import numpy as np

#reading Dataset
dataset=pd.read_csv("a2.csv")
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,5]

#Perform Label encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

X=X.apply(le.fit_transform)
print(X)

from sklearn.tree import DecisionTreeClassifier
regressor=DecisionTreeClassifier()
regressor.fit(X.iloc[:,1:5],y)

#Predict value for the given Expression
X_in=np.array([1,1,0,0])
y_pred=regressor.predict([X_in])
print("Prediction:", y_pred)
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

dot_data=StringIO()

export_graphviz(regressor,out_file=dot_data,filled=True,rounded=True,special_characters=True)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.write_png('tree.png'))







########KMEANS
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
