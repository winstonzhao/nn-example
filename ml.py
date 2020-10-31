from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

# Example dataset
df = pd.read_csv("./test.csv")
df = df[["BldgType", "HouseStyle", "GarageCars", "YearBuilt", "KitchenAbvGr", "YearRemodAdd", "YrSold", "GarageYrBlt"]]


# How to encode categorical data
df = df.join(pd.get_dummies(df['BldgType']))
df = df.drop('BldgType', axis=1)
df = df.join(pd.get_dummies(df['HouseStyle']))
df = df.drop('HouseStyle', axis=1)

# Some required preprocessing
df = df.fillna(0)
X = df.to_numpy()

# Train Model
nn = NearestNeighbors(n_neighbors=2, radius=0.4).fit(X)

# Input Data in order of most recent to least recent
inpt = [X[0], X[1], X[2], X[3], X[4]]
inpt_idx = [0, 1, 2, 3, 4]


# Get most similar houses to this random example
output = nn.kneighbors(inpt, 5)

distances = output[0]

# Apply recency bias, parameter = 1.5
for i in range(0, 5):
    distances[i] = distances[i] * (1.5 ** i)

index_flat = output[1].flatten()



#  # Returns [[ 819 1150  558    0  975]]
#  # This means that the elements in the variable X, at the indices in the list above are the most similar to that which we passed in.