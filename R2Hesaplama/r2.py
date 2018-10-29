#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 20:07:43 2018

@author: ozcanyureklioglu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

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


#dogrusal olmayan regresyon(polinom)/2.dereceden
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


#Standard Scaler
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcekli=sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli=sc2.fit_transform(Y)


#SVR machine/Destek Vektör Makineleri
from sklearn.svm import SVR
sv_reg=SVR(kernel='rbf')
sv_reg.fit(x_olcekli,y_olcekli)

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
 
#Karar Agaci/Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
plt.title("Karar Agaci")
plt.scatter(X,Y,color='red')
plt.plot(X,r_dt.predict(X),color='blue')
plt.show()

#Rassal Orman/Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y)
plt.scatter(X,Y,color='red')
plt.plot(X,rf_reg.predict(X),color='blue')
plt.show()

#R2 degerleri /Algoritma Karsilastirma
print("Linear Regression r2 hesabı:")
print(r2_score(Y,lin_reg.predict(X)),"\n")

print("Polynomial Regression r2 hesabı(degree:2):")
print(r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(X))),"\n")

print("Polynomial Regression r2 hesabı(degree:4):")
print(r2_score(Y,lin_reg3.predict(poly_reg3.fit_transform(X))),"\n")

print("SVR r2 hesabı:")
print(r2_score(y_olcekli,sv_reg.predict(x_olcekli)),"\n")

print("Decision Tree Regressor r2 hesabı:")
print(r2_score(Y,r_dt.predict(X)),"\n")

print("Random Forest Regressor r2 hesabı:")
print(r2_score(Y,rf_reg.predict(X)))






