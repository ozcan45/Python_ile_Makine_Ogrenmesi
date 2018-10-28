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

#verilerin yuklenmesi
veriler=pd.read_csv("maaslar.csv")

#verilerin frame dilimleme(slice)
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

#frame verilerin numPY dizi dönüşümü
X=x.values
Y=y.values


#linear regression dogrusal regresyon
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)


#dogrusal olmayan regreyson(polinom)/2.dereceden
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,Y)


#dogrusal olmayan regresyon(polinom)/4.dereceden
poly_reg3=PolynomialFeatures(degree=4)
x_poly3=poly_reg3.fit_transform(X)
lin_reg3=LinearRegression()
lin_reg3.fit(x_poly3,Y)




#gorsellestirme
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X),color='blue')
plt.show()

plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='orange')
plt.show()

plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)),color='blue')
plt.show()
 






#destek vektor makineleri /Support Vector Machine

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcekli=sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli=sc2.fit_transform(Y)

from sklearn.svm import SVR
sv_reg=SVR(kernel='rbf')
sv_reg.fit(x_olcekli,y_olcekli)
plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,sv_reg.predict(x_olcekli),color='blue')
plt.show()












