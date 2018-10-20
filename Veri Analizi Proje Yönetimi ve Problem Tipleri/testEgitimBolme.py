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
veriler=pd.read_csv("eksikveriler.csv")
#print(veriler)
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
yas=veriler.iloc[:,1:4].values
imputer=imputer.fit(yas[:,1:4])
yas[:,1:4]=imputer.transform(yas[:,1:4])
#print(yas)
#------------------------------------------------------





#kategorik veriler ülkeleri sayısal veriye dönüştürme

ulke=veriler.iloc[:,0:1].values
le=LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])


ohe=OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()

#print('Encoding edilen ulke verisi:',ulke)
#-----------------------------------------------------------




#verilerin birleştirilmesi

sonucUlke=pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
#print(sonucUlke)
sonucBoyKiloYas=pd.DataFrame(data=yas,index=range(22),columns=['boy','kilo','yas'])
#print(sonucBoyKiloYas)
cinsiyet=veriler.iloc[:,-1:].values
sonucCinsiyet=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
sonuc=pd.concat([sonucUlke,sonucBoyKiloYas,sonucCinsiyet],axis=1)
sonucTest=pd.concat([sonucUlke,sonucBoyKiloYas],axis=1)
print(sonucTest)
#print(sonuc)
#---------------------------------------------------------------






#test egitimi ve bölme

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test=train_test_split(sonucTest,sonucCinsiyet,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)















