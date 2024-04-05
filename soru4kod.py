#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 23:36:39 2024

@author: seckinaktas
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


df = pd.read_excel("normalized_veriler.xlsx") 


X = df.drop(columns=['Class variable (0 or 1)'])  
y = df['Class variable (0 or 1)']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


clf = DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=3, min_samples_leaf=6)


clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Classification Report:\n", class_report)








clf = DecisionTreeClassifier()
clf.fit(X, y)


plt.figure(figsize=(15, 10))  
plot_tree(clf, feature_names=X.columns, class_names=y.unique(), filled=True, rounded=True)
plt.show()


