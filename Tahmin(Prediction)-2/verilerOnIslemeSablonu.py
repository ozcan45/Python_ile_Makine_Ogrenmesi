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




#eksikVeriler
veriler=pd.read_csv("veriler.csv")
#print(veriler)
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder

#------------------------------------------------------

yas=veriler.iloc[:,1:4].values

#print(yas)


#kategorik veriler ülkeleri sayısal veriye dönüştürme

ulke=veriler.iloc[:,0:1].values
le=LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
#print(ulke)

ohe=OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()
#print(ulke)
#print('Encoding edilen ulke verisi:',ulke)
#-----------------------------------------------------------


#cinsiyet alanı numarik veriye çevirme

c=veriler.iloc[:,-1:].values
le=LabelEncoder()
c[:,0] = le.fit_transform(c[:,0])
#print(c)

ohe=OneHotEncoder(categorical_features='all')
c=ohe.fit_transform(c).toarray()
#print(c)
#-------------------------------------------------------------



#verilerin birleştirilmesi

sonucUlke=pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
#print(sonucUlke)
sonucBoyKiloYas=pd.DataFrame(data=yas,index=range(22),columns=['boy','kilo','yas'])
#print(sonucBoyKiloYas)

sonucCinsiyet=pd.DataFrame(data=c[:,:1],index=range(22),columns=['cinsiyet'])
sonuc=pd.concat([sonucUlke,sonucBoyKiloYas,sonucCinsiyet],axis=1)
sonucTest=pd.concat([sonucUlke,sonucBoyKiloYas],axis=1)
#print(sonucTest)
#print(sonuc)
#---------------------------------------------------------------






#test egitimi ve bölme

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test=train_test_split(sonucTest,sonucCinsiyet,test_size=0.33,random_state=0)

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
boy=sonuc.iloc[:,3:4].values
sol=sonuc.iloc[:,:3]
sag=sonuc.iloc[:,4:]
veri=pd.concat([sol,sag],axis=1)

x_train,x_test,y_train,y_test=train_test_split(veri,boy,test_size=0.33,random_state=0)

r2=LinearRegression()
r2.fit(x_train,y_train)
y_pred=r2.predict(x_test)
#print(y_pred)
import statsmodels.formula.api as sm
X=np.append(arr=np.np.ones((22,1)).astype(int),values=veri ,axis=1)
X_l=veri.iloc[:,[0,1,2,3,4,5]].values






