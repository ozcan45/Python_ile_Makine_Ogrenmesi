#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 20:07:43 2018

@author: ozcanyureklioglu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#------------------------------------------------------
#veriYukleme
"""
veriler=pd.read_csv("veriler.csv")

print(veriler)
"""

#------------------------------------------------------
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder

#verileri yukleme
veriler=pd.read_csv("odev_tenis.csv")
#print(veriler)


#------------------------------------------------------
#numarik veri cevirme

#print(veriler2)

veriler2=veriler.apply(LabelEncoder().fit_transform)

le=OneHotEncoder(categorical_features="all")
c=veriler2.iloc[:,:1]
c=le.fit_transform(c).toarray()

#verilerin birleştirilmesi
havadurumu=pd.DataFrame(data=c,index=range(14),columns=['o','r','s'])
sonveriler=pd.concat([havadurumu,veriler.iloc[:,1:3]],axis=1)
sonveriler2=pd.concat([veriler2.iloc[:,-2:],sonveriler],axis=1)

#print(sonuc)
#---------------------------------------------------------------


#test egitimi ve bölme

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test=train_test_split(sonveriler2.iloc[:,:-1],sonveriler2.iloc[:,-1:],test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
#X_train=sc.fit_transform(x_train)
#X_test=sc.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
#linear model
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print("ilk tahmin:",y_pred)


x_train,x_test,y_train,y_test=train_test_split(sonveriler2.iloc[:,:-1],sonveriler2.iloc[:,-1:],test_size=0.33,random_state=0)

r2=LinearRegression()
r2.fit(x_train,y_train)
y_pred=r2.predict(x_test)
#print(y_pred)
#ilk p değeri eleme yok
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((14,1)).astype(int),values=sonveriler2.iloc[:,:-1] ,axis=1)
X_l=sonveriler2.iloc[:,[0,1,2,3,4,5]].values
r_ols=sm.OLS(endog=sonveriler2.iloc[:,-1:],exog=X_l).fit()
print(r_ols.summary())
#ilk p değeri regression hesaplama,tahmin etme

x_train,x_test,y_train,y_test=train_test_split(sonveriler2.iloc[:,:-1],sonveriler2.iloc[:,-1:],test_size=0.33,random_state=0)
r2=LinearRegression()
r2.fit(x_train,y_train)
y_pred=r2.predict(x_test)
print("ikinci tahmin: ",y_pred)

#ikinci geri eleme ve regression hesaplama,tahmin etme

sonveriler2=sonveriler2.iloc[:,1:]
X=np.append(arr=np.ones((14,1)).astype(int),values=sonveriler2.iloc[:,:-1] ,axis=1)
X_l=sonveriler2.iloc[:,[0,1,2,3,4]].values
r_ols=sm.OLS(endog=sonveriler2.iloc[:,-1:],exog=X_l).fit()
print(r_ols.summary())
#Elemeden sonraki tahmin 
x_train,x_test,y_train,y_test=train_test_split(sonveriler2.iloc[:,:-1],sonveriler2.iloc[:,-1:],test_size=0.33,random_state=0)

r2=LinearRegression()
r2.fit(x_train,y_train)
y_pred=r2.predict(x_test)
print("üçüncü tahmin: ",y_pred)




