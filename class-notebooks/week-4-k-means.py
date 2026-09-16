# coding: utf-8
# import libraries
import pandas as pd
import matplotlib.pyplot as plt
# load up CSV
df = pd.read_csv("people.csv")

# create an initial plot to see the data
plt.scatter(df["age"], df["annual_income"])
plt.xlabel("Age")
plt.ylabel("Annual income")
plt.show()
# import KMeans from scikit learn
from sklearn.cluster import KMeans

# map our inputs to X
X = df[["age", "annual_income"]]

# define our model with 3 clusters
model = KMeans(n_clusters=3, random_state=42)

# train the model
model.fit(X)

# create a column with our new clusters (as labels)
df["cluster"] = model.labels_
df
# create a scatter plot
plt.scatter( df["age"], df["annual_income"], c=df["cluster"])
plt.xlabel("Age")
plt.ylabel("Annual income")
plt.show()
# get some information about our clusters
df.groupby("cluster")[["age", "annual_income"]].mean()
# try with a new value for k=2
model = KMeans(n_clusters=2, random_state=42)
model.fit(X)
df["cluster"] = model.labels_
df
plt.scatter( df["age"], df["annual_income"], c=df["cluster"])
plt.xlabel("Age")
plt.ylabel("Annual income")
plt.show()
df.groupby("cluster")[["age", "annual_income"]].mean()
# include scaling to regularize values
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
model = KMeans(n_clusters=3, random_state=42)

model.fit(X_scaled)

df["cluster"] = model.labels_
df
plt.scatter( df["age"], df["annual_income"], c=df["cluster"])
plt.xlabel("Age")
plt.ylabel("Annual income")
plt.show()
X
X_scaled
