from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

df = pd.read_csv("./test.csv")
df = df[["BldgType", "HouseStyle", "GarageCars", "YearBuilt", "KitchenAbvGr"]]

df = df.join(pd.get_dummies(df['BldgType']))
df = df.drop('BldgType', axis=1)

df = df.join(pd.get_dummies(df['HouseStyle']))
df = df.drop('HouseStyle', axis=1)
df = df.fillna(0)
X = df.to_numpy()
kmeans = NearestNeighbors(n_neighbors=2, radius=0.4).fit(X)

print(kmeans.kneighbors([[1.000e+00, 1.961e+03, 1.000e+00, 1.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
 0.000e+00, 0.000e+00, 0.000e+00, 1.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
 0.000e+00]], 5, False))