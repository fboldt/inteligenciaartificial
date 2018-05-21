# -*- coding: utf-8 -*-
"""
Created on Tue May 15 08:01:38 2018

@author: francisco
"""
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,RepeatedStratifiedKFold
from sklearn import datasets, metrics
import numpy as np
import time

clfs = {"SVM": svm.SVC(),
        "SVMs": Pipeline([('scaler', StandardScaler()),
                          ('SVM', svm.SVC())]),
        "SVMsgs": Pipeline([('scaler', StandardScaler()),
                          ('SVMgs', GridSearchCV(svm.SVC(),
                              {'kernel':('linear', 'rbf'),
                               'C':[0.001, 1, 1000]},'f1_macro'))]),
        }

dataset = datasets.load_iris()#datasets.load_wine()#
X = dataset["data"]
y = dataset["target"]

rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)
performances={}
tot = time.time()
for train, test in rkf.split(X,y):
    for key,clf in clfs.items():
        if key not in performances:
            performances[key] = {'f1_score': [],
                        'accuracy': [],
                        'traitime': [],
                        'testtime': []}
        ti = time.time()
        clf.fit(X[train],y[train])
        trt = time.time()-ti
        ti = time.time()
        predicted = clf.predict(X[test])
        tet = time.time()-ti
        performances[key]['f1_score'].append(
                metrics.f1_score(y[test], predicted,average='macro'))
        performances[key]['accuracy'].append(
                metrics.accuracy_score(y[test], predicted))
        performances[key]['traitime'].append(trt)
        performances[key]['testtime'].append(tet)
        
print(time.time()-tot)
for key,perf in performances.items():
    print(key,
          '\t', 'f1s:{:.4f}'.format(np.mean(perf['f1_score'])), 
          'acc:{:.4f}'.format(np.mean(perf['accuracy'])),
          'trt:{:.4f}'.format(np.mean(perf['traitime'])),
          'tet:{:.4f}'.format(np.mean(perf['testtime'])))
