import numpy as np
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score

#Read the input file
df = pd.read_csv("../data/master.csv")

# To convert category into numerical labels
columns = df.select_dtypes(['category']).columns
df[columns] = df[columns].apply(lambda fx: fx.cat.codes)
df.dtypes

#Fit the model and predict the data
x = df.drop('suicides_no', axis=True)
x = df.drop('year', axis=True)
y = df['suicides_no']
kmeans = KMeans(n_clusters=2)
kmeans.fit(x)
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=600,
    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=0, tol=0.0001, verbose=0)
y_kmeans = kmeans.predict(x)
x, y_kmeans = make_blobs(n_samples=600, centers=2, cluster_std=0.60, random_state=0)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x[:,0], x[:,1], c=y_kmeans, cmap='cool')

#Print the score
print(silhouette_score(x, y_kmeans))