
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.cluster import AffinityPropagation
from sklearn import metrics



karate_data_set_list=[
[464.197614991482 , 359.623509369676],
[552.681431005111 , 306.533219761499],
[567.218057921636 , 267.347529812606],
[653.173764906303 , 313.485519591141],
[512.863713798978 , 275.563884156729],
[531.824531516184 , 226.897785349233],
[469.885860306644 , 166.855195911414],
[400.362862010222 , 243.330494037479],
[410.475298126065 , 312.853492333901],
[378.241908006814 , 339.398637137990],
[381.402044293015 , 185.183986371380],
[367.497444633731 , 214.257240204429],
[330.839863713799 , 262.291311754685],
[269.533219761499 , 265.451448040886],
[298.606473594549 , 209.833049403748],
[275.221465076661 , 180.127768313458],
[237.299829642249 , 224.369676320272],
[198.114139693356 , 259.763202725724],
[128.591141396934 , 197.824531516184],
[141.231686541738 , 182.655877342419],
[191.793867120954 , 161.166950596252],
[204.434412265758 , 118.821124361158],
[244.252129471891 , 127.669505962521],
[182.313458262351 , 23.3850085178875],
[237.299829642249 , 15.8006814310050],
[194.321976149915 , 84.0596252129471],
[119.110732538331 , 91.6439522998296],
[61.5962521294720 , 123.245315161840],
[9.77001703577520 , 109.340715502555],
[39.4752981260648 , 167.487223168654],
[24.3066439522999 , 199.088586030664],
[35.0511073253834 , 228.793867120954],
[55.2759795570699 , 256.603066439523],
[95.0936967632028 , 274.931856899489]
]

X=np.array(karate_data_set_list)
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

