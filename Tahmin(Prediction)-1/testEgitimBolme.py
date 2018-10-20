#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 20:07:43 2018

@author: ozcanyureklioglu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




#eksikVeriler
veriler=pd.read_csv("satislar.csv")
aylar=veriler[['Aylar']]
satislar=veriler[['Satislar']]
print(satislar)
print(aylar)

#test egitimi ve bölme

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test=train_test_split(aylar,satislar,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)















