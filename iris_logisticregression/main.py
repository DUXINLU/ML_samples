# coding=UTF-8
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

# load iris dataset
X = load_iris().data
y = load_iris().target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
LR_model = LogisticRegression(penalty='l2', solver='newton-cg', multi_class='multinomial', verbose=1)
LR_model.fit(X_train, y_train)

y_predict = LR_model.predict(X_test)
print(y_test.shape[0])
accuracy = metrics.accuracy_score(y_test, y_predict)
print(LR_model.score(X_train, y_train), LR_model.score(X_test, y_test), accuracy)

plt.plot(y_test)
plt.plot(y_predict)
plt.show()
