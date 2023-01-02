7. Apply EM algorithm to cluster a set of data stored in a .CSV file. Use the same data set 
for clustering using k-Means algorithm. Compare the results of these two algorithms and 
comment on the quality of clustering. You can add Java/Python ML library classes/API in 
the program. 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
l1 = [0,1,2]
def rename(s):
 l2 = []
 for i in s:
 if i not in l2:
 l2.append(i)
 for i in range(len(s)):
 pos = l2.index(s[i])
 s[i] = l1[pos]
 return s
iris = datasets.load_iris()
X = pd.DataFrame(iris.data,columns 
=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width'] )
y = pd.DataFrame(iris.target,columns = ['Targets'])
def graph_plot(l,title,s,target):
 plt.subplot(l[0],l[1],l[2])
 if s==1:
 plt.scatter(X.Sepal_Length,X.Sepal_Width, c=colormap[target], s=40)
 else:
 plt.scatter(X.Petal_Length,X.Petal_Width, c=colormap[target], s=40)
 plt.title(title)
 
plt.figure()
colormap = np.array(['red', 'lime', 'black'])
graph_plot([1, 2, 1],'sepal',1,y.Targets)
graph_plot([1, 2, 2],'petal',0,y.Targets)
plt.show()
def fit_model(modelName): 
 model = modelName(3)
 model.fit(X)
 plt.figure()
 colormap = np.array(['red', 'lime', 'black'])
 graph_plot([1, 2, 1],'Real Classification',0,y.Targets)
 if modelName == KMeans:
 m = 'Kmeans’
 else:
 m = 'Em'
 y1 = model.predict(X)
 graph_plot([1, 2, 2],m,0,y1)
 plt.show()
 km = rename(y1)
 print("\nPredicted: \n", km)
 print("Accuracy ",sm.accuracy_score(y, km))
 print("Confusion Matrix ",sm.confusion_matrix(y, km))
fit_model(KMeans)
fit_model(GaussianMixture)
OUTPUT:
IRIS DATA : [[5.1 3.5 1.4 0.2]
[4.9 3. 1.4 0.2]
[4.7 3.2 1.3 0.2]
[4.6 3.1 1.5 0.2]
[5. 3.6 1.4 0.2]
[5.4 3.9 1.7 0.4]
[4.6 3.4 1.4 0.3]
[5. 3.4 1.5 0.2]
[4.4 2.9 1.4 0.2
