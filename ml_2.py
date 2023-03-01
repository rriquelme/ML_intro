from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn # need to remove...

def make_forge():
    X, y = make_blobs(centers=2, random_state=4, n_samples=30)
    y[np.array([7, 27])] = 0
    mask = np.ones(len(X), dtype=np.bool_)
    mask[np.array([0, 1, 5, 26])] = 0
    X, y = X[mask], y[mask]
    return X, y


# generate dataset
X, y = make_forge()
print()
X_df = pd.DataFrame(X,columns=["a","b"])
X_df.plot(x = 'a',y='b',c = y.tolist(),kind='scatter')
plt.legend(["Class 0", "Class 1"], loc=1)
plt.legend(["Class 1", "Class 2"], loc=1)
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()
print("X.shape: {}".format(X.shape))