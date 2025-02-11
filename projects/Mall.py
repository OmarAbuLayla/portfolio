import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


df = pd.read_csv("Mall_Customers.csv")
samples = df.drop(columns=["Genre", "CustomerID", "Age"])
scaler = StandardScaler()
samples_2 = scaler.fit_transform(samples)
listt = []

for k in range(2,11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(samples_2)
    listt.append(model.inertia_)

plt.plot(range(2,11), listt)
plt.show() 

plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"])
plt.show()

 # The plot tells us that there is 6 diff clusters to group.

model = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = model.fit_predict(samples_2)
customers = samples.copy()
customers["Cluster"] = df["Cluster"]
customers = customers.groupby("Cluster").mean()
print(customers)


