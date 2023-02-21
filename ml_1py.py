from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


iris_dataset = load_iris()

# Keys
print(iris_dataset.keys())
# description
#print(iris_dataset['DESCR'])

# to data frame:
iris_df = pd.DataFrame(iris_dataset['data'],columns = iris_dataset['feature_names'])
iris_df['target'] = pd.Series(iris_dataset['target'])
print(iris_dataset['target'])
print(iris_df.head())
print(iris_df.describe())

# Split test and train dataset
train, test = train_test_split(iris_df,random_state=0)
print(test.shape)
print(train.shape)

#Scatter Plot
#iris_df.plot(x = iris_dataset['target'][0], y = iris_dataset['target'][1],kind='scatter')
#pd.plotting.scatter_matrix(iris_df.drop(columns=['target']),hist_kwds={'bins': 20},c = iris_df['target']) # Same as below
#pd.plotting.scatter_matrix(iris_df[iris_dataset['feature_names']],hist_kwds={'bins': 20},c = iris_df['target'])
#plt.show()

# k nearest neighbors with a single neighbor
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train[iris_dataset['feature_names']].values,train['target'].values)

prediction = knn.predict(np.array([[5,2.9,1,0.2]]))
print(prediction)
print(iris_dataset['target_names'][prediction])

# Evaluating Model by hand

result_prediction = knn.predict(test[iris_dataset['feature_names']].values)
print(result_prediction)

print("Final Score: {:.2f}".format(np.mean(test['target']==result_prediction)))

# Evaluating model with built-in function

print(knn.score(test[iris_dataset['feature_names']].values,test['target'].values))


