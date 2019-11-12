import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from DecisionTree import DecisionTree
from GradientBoostingClassifier import GradientBoostingClassifier
from Model import accuracy_score


def fit(self, X, y):
    y_pred = np.full(np.shape(y), np.mean(y, axis=0))
    for i in self.bar(range(self.n_estimators)):
        gradient = self.loss.gradient(y, y_pred)
        self.trees[i].fit(X, gradient)
        update = self.trees[i].predict(X)
        # Update y prediction
        y_pred -= np.multiply(self.learning_rate, update)


def predict(self, X):
    y_pred = np.array([])
    # Make predictions
    for tree in self.trees:
        update = tree.predict(X)
        update = np.multiply(self.learning_rate, update)
        y_pred = -update if not y_pred.any() else y_pred - update

    if not self.regression:
        # Turn into probability distribution
        y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
        # Set label to the value that maximizes probability
        y_pred = np.argmax(y_pred, axis=1)
    return y_pred


dataset = pd.read_csv('mushrooms.csv')

print(dataset.head())

#Define the column names
dataset.columns = ['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing',
'gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring',
'stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population',
'habitat']

for label in dataset.columns:
    dataset[label] = LabelEncoder().fit(dataset[label]).transform(dataset[label])

le = LabelEncoder()
dataset = dataset.apply(le.fit_transform)

X = dataset.drop(['class'], axis=1)
Y = dataset['class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


X_train = X_train.to_numpy()
Y_train = Y_train.to_numpy()
X_test = X_test.to_numpy()
Y_test = Y_test.to_numpy()
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print ("-- Gradient Boosting Classification --")

clf = GradientBoostingClassifier()
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)

print ("Accuracy:", accuracy)
