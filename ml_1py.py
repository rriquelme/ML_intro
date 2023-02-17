from sklearn.datasets import load_iris

iris_dataset = load_iris()
# Keys
print(iris_dataset.keys())
# description
#print(iris_dataset['DESCR'])

print(iris_dataset['target'])
