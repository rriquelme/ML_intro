from sklearn.datasets import load_iris
import pandas as pd
iris_dataset = load_iris()
# Keys
print(iris_dataset.keys())
# description
#print(iris_dataset['DESCR'])
# to data frame:

iris_df = pd.DataFrame(iris_dataset['data'],columns = iris_dataset['feature_names'])
iris_df['target'] = pd.Series(iris_dataset['target'])
#print(iris_dataset['target'])

print(iris_df.head())

print(iris_df.describe())