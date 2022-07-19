from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
iris_data = load_iris()
input_train, input_test = train_test_split(iris_data.data, random_state=0)
output_train, output_test = train_test_split(iris_data.target, random_state=0)

input_new=np.array([[6.0,3.23,4.5,2.0]])
mytree=tree.DecisionTreeClassifier()
model=mytree.fit(input_train,output_train)
kq=model.predict(input_new)
print(kq)
print(model.score(input_test,output_test))