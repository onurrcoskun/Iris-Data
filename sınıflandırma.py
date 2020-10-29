# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:13:03 2020

@author: Onur
"""

#data yükleme
from sklearn.datasets import load_iris
dataSet = load_iris()

features= dataSet.data
labels= dataSet.target
labelsNames= list(dataSet.target_names)
featureNames =dataSet.feature_names

print([labelsNames[i]for i in labels[47:52]])
print(featureNames)

#data analiz
import pandas as pd

print(type(features))

featuresDF=pd.DataFrame(features)
featuresDF.columns = featureNames

print(type(featuresDF))
print(featuresDF.describe())
print(featuresDF.info())

#data görselleştirme
featuresDF.plot()

#data model
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()

#data test
import numpy as np
from sklearn.model_selection import train_test_split
X= features
y= labels

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
print((X_test))

#eğitim modeli
clf.fit(X_train,y_train)

#test modeli

accuracy=clf.score(X_test,y_test)
print("accuracy on test data{:.2}%" .format(accuracy))

#iyileştirme
from joblib import dump, load
filename="myfirst.joblib"
dump(clf,filename)