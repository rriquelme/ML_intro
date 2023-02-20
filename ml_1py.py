from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
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
pd.plotting.scatter_matrix(iris_df[iris_dataset['feature_names']],hist_kwds={'bins': 20},c = iris_df['target'])
plt.show()