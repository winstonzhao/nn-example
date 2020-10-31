from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

# Example dataset
df = pd.read_csv("./test.csv")
df = df[["BldgType", "HouseStyle", "GarageCars", "YearBuilt", "KitchenAbvGr"]]

# How to encode categorical data
df = df.join(pd.get_dummies(df['BldgType']))
df = df.drop('BldgType', axis=1)
df = df.join(pd.get_dummies(df['HouseStyle']))
df = df.drop('HouseStyle', axis=1)

# Some required preprocessing
df = df.fillna(0)
X = df.to_numpy()

# Train Model
kmeans = NearestNeighbors(n_neighbors=2, radius=0.4).fit(X)

# Get most similar houses to this random example
print(kmeans.kneighbors([[1.000e+00, 1.961e+03, 1.000e+00, 1.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
 0.000e+00, 0.000e+00, 0.000e+00, 1.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
 0.000e+00]], 5, False))

 # Returns [[ 819 1150  558    0  975]]
 # This means that the elements in the variable X, at the indices in the list above are the most similar to that which we passed in.