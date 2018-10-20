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




#verileri yukleme
veriler=pd.read_csv("odev_tenis.csv")
#print(veriler)
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder

#------------------------------------------------------

#print(veriler)
termo=veriler.iloc[:,1:2].values
humidity=veriler.iloc[:,2:3].values
#print(termo)


#kategorik veriler hava durumu alanı numarik çevirim

outlook=veriler.iloc[:,0:1].values
le=LabelEncoder()
outlook[:,0] = le.fit_transform(outlook[:,0])
#print(outlook)

ohe=OneHotEncoder(categorical_features='all')
outlook=ohe.fit_transform(outlook).toarray()
#print(outlook)

#-----------------------------------------------------------


#windy alanı numarik veriye çevirme

windy=veriler.iloc[:,3:4].values
le=LabelEncoder()
windy[:,0] = le.fit_transform(windy[:,0])
#print(windy)

ohe=OneHotEncoder(categorical_features='all')
windy=ohe.fit_transform(windy).toarray()
#print(windy)
#-------------------------------------------------------------




#oyun alanı numarik veriye çevirme

play=veriler.iloc[:,-1:].values
le=LabelEncoder()
play[:,0] = le.fit_transform(play[:,0])
#print(play)

ohe=OneHotEncoder(categorical_features='all')
play=ohe.fit_transform(play).toarray()
#print(play)
#-------------------------------------------------------------





#verilerin birleştirilmesi

sonuc1=pd.DataFrame(data=outlook,index=range(14),columns=['overcast','rainy','sunny'])
sonuc2=pd.DataFrame(data=termo,index=range(14),columns=['temperature'])
sonuc3=pd.DataFrame(data=windy[:,:1],index=range(14),columns=['windy'])
sonuc4=pd.DataFrame(data=play[:,:1],index=range(14),columns=['play'])

sonuc=pd.concat([sonuc1,sonuc2,sonuc3,sonuc4],axis=1)


#print(sonuc)
#---------------------------------------------------------------






#test egitimi ve bölme

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test=train_test_split(sonuc,humidity,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
#X_train=sc.fit_transform(x_train)
#X_test=sc.fit_transform(x_test)

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
#print(y_pred)

#boy tahmin etme regresion'u
#boy=sonuc.iloc[:,3:4].values
#sol=sonuc.iloc[:,:3]
#sag=sonuc.iloc[:,4:]
#veri=pd.concat([sol,sag],axis=1)

'''x_train,x_test,y_train,y_test=train_test_split(veri,boy,test_size=0.33,random_state=0)

r2=LinearRegression()
r2.fit(x_train,y_train)
y_pred=r2.predict(x_test)
#print(y_pred)
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((22,1)).astype(int),values=veri ,axis=1)
X_l=veri.iloc[:,[0,1,2,3,4,5]].values
r_ols=sm.OLS(endog=boy,exog=X_l).fit()
print(r_ols.summary())
'''




