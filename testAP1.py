
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.cluster import AffinityPropagation
from sklearn import metrics



file_name= '2DKarateEmbeddings'
with open(file_name, 'r') as f:
    next(f)
    data = f.readlines()

x2=[]
emb_dict={}
content = [x.strip() for x in data] 
N=len(content)
Z=[]
d=2
D=2

for row in content:
    row_t = row.split()
    x1=[]
    for i in range(D):
      x1.append(float(row_t[i+1]))
    x2.append(x1)
    Z.append(row_t[0])

X=np.array(x2)
	
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1])
plt.title("Karate dataset")	   



# Compute Affinity Propagation
af = AffinityPropagation(preference=-50).fit(X)
cluster_centers_indices = af.cluster_centers_indices_

labels = af.labels_

n_clusters_ = len(cluster_centers_indices)


plt.subplot(222)

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('The estimated number of clusters: %d' % n_clusters_)
plt.show()

