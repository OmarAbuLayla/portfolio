'''
Utilize your unsupervised learning skills to clusters in the penguins dataset!

Import, investigate and pre-process the "penguins.csv" dataset.
Perform a cluster analysis based on a reasonable number of clusters and collect the average values for the clusters. 
The output should be a DataFrame named stat_penguins with one row per cluster that shows the mean of the- 
original variables (or columns in "penguins.csv") by cluster. 
stat_penguins should not include any non-numeric columns.

'''

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("penguins.csv")
samples = df.drop(columns="sex")
scaler = StandardScaler()
samples_Scaled = scaler.fit_transform(samples)


inertias = []
rrange = range(1,11)

for k in rrange:
    model = KMeans(n_clusters=k)
    model.fit(samples)
    inertias.append(model.inertia_)

plt.plot(rrange, inertias)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Clusters')
plt.show()
