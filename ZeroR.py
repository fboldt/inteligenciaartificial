# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from scipy.spatial import distance


class ZeroR(BaseEstimator, ClassifierMixin):    
    def __init__(self):
        pass    
    def fit(self, X, y):
        self.labels, self.counts = np.unique(y, return_counts=True)
    def predict(self, X, y=None):
        index = np.argmax(self.counts)
        response = self.labels[index]
        return [response for x in X]    
    def score(self, X, y):
        return(sum(self.predict(X)==y)/len(y)) 

class MeanClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass    
    def fit(self, X, y):
        self.labels = list(set(y))
        self.means = np.zeros(shape=(len(self.labels),X.shape[1]))
        for i,label in enumerate(self.labels):
            idx = y==label
            self.means[i] = sum(X[idx])/len(X[idx]) #Mean for each label
    def predict(self, X, y=None):        
        return [self.labels[np.argmin(np.mean(np.absolute(x-self.means), axis=1))] for x in X]            
    def score(self, X, y):
        return(sum(self.predict(X)==y)/len(y))

class CentroidClassifier(BaseEstimator, ClassifierMixin):    
    def __init__(self):
        pass    
    def fit(self, X, y):
        self.labels = list(set(y))
        self.median = np.zeros(shape=(len(self.labels),X.shape[1]))
        for i,label in enumerate(self.labels):
            idx = y==label
            self.median[i] = np.median(X[idx], axis=0)
    def predict(self, X, y=None):
        idxs = np.argmin(distance.cdist(X,self.median), axis=1)
        return [self.labels[i] for i in idxs]    
    def score(self, X, y):
        return(sum(self.predict(X)==y)/len(y)) 
