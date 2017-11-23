# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:52:08 2017

@author: 2304946
"""

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt


pipeline = Pipeline([
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', PolynomialFeatures(degree=3, include_bias=False)),
    ('std_scaler', StandardScaler()),
])


class Patterns:
    def getPattern(self,strdate,index=0):
        date = datetime.strptime(str(strdate), '%Y-%m-%d')
        pattern = pd.DataFrame(
                np.matrix([date.year,date.month,date.day]).reshape(1, 3),
                columns=['year','month','day'],
                index=[index])
        return pattern,date
        
#'''
csvfile = "csv/dollar.csv"
'''
csvfile = "csv/iphone.csv"
#'''    
data = pd.read_csv(csvfile)

regr = SGDRegressor()
dates = []
yactu = []
ypred = []
values = np.array([])
genpat = Patterns()
patterns = pd.DataFrame()
nip = 26

for i in range(nip):
    pattern,date = genpat.getPattern(data['Date'].loc[i],i)
    patterns = patterns.append(pattern)
    dates.append(date)
    value = data['Value'].loc[i]
    values = np.append(values, value)

pipeline.fit(patterns)
regr.partial_fit(pipeline.transform(patterns),values)

for i in range(nip):
    pattern = patterns.iloc[[i]]
    value = data['Value'].loc[i]
    yactu.append(value)
    ypred.append(regr.predict(pipeline.transform(pattern)))

for i in range(nip,len(data)):
    pattern,date = genpat.getPattern(data['Date'].loc[i],i)
    patterns = patterns.append(pattern)
    dates.append(date)
    value = data['Value'].loc[i]
    yactu.append(value)
    ypred.append(regr.predict(pipeline.transform(pattern)))
    regr.partial_fit(pipeline.transform(pattern),[value],[1])

for i in range(len(data),len(data)+nip):
    date = date+timedelta(days=7)
    pattern,date = genpat.getPattern(date.strftime('%Y-%m-%d'),i)
    patterns = patterns.append(pattern)
    dates.append(date)
    ypred.append(regr.predict(pipeline.transform(pattern)))

fig_size = plt.rcParams["figure.figsize"]
factor=1
fig_size[0] = 8*factor
fig_size[1] = 6*factor
plt.rcParams["figure.figsize"] = fig_size
fi = 0
plt.plot(dates[fi:len(data)],yactu[fi:len(data)])
plt.plot(dates[fi:],ypred[fi:])
