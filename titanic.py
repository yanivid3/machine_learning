#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:25:18 2017

@author: Shiradvd
"""


import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing

#set your working_dir
working_dir = "/Users/Shiradvd/Desktop/ML/TiatanicDemo/data"

df = pd.read_excel(working_dir+"/titanic.xls")

df.head()

df.drop(['body','name'],1, inplace=True)

df.convert_objects(convert_numeric=True)

#filling empty as 0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
df.fillna(0, inplace=True)
#print(df.head())

def handle_non_numeric_data(df):
    columns = df.columns.values
    
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
                    
            df[column] = list(map(convert_to_int,df[column]))
            
            
    return df

df = handle_non_numeric_data(df)
#print(df.head())

#remove the survived column - this is what we want to explore
X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf= KMeans(n_clusters=2)
clf.fit(X)

correct = 0.0

for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct +=1
     
    
from decimal import Decimal    
    
print(y)    
print(correct)
print(len(X))    

amount_correct = correct/len(X)
print(amount_correct)       
    
    