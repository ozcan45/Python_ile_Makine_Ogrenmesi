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
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
yas=veriler.iloc[:,1:4].values
imputer=imputer.fit(yas[:,1:4])
yas[:,1:4]=imputer.transform(yas[:,1:4])
print(yas)




