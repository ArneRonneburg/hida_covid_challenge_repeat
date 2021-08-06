# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 20:17:47 2021

@author: Arne
"""

import os
import time
import sys
import numpy as np
import sklearn as sk
import scipy as scp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import KNNImputer as KNN
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer as II
path=r"C:\Users\Arne\Documents\DataScience\hida_covid_challenge_repeat/"
train=pd.read_csv(path + "trainSet.txt", index_col=None)
test=pd.read_csv(path + "testSet.txt", index_col=None)
####define measures for the imputation quality. Idea 1: they have a similar mean and variance 

#for a given column, calculate the mean and the variance
moments={}
for i in train.columns[3:-1]:##exclude patient ID, patient image, hospital, prognosis
    moments[i]=[train[i].describe()[1], train[i].describe()[2]]

def imput_accur_moments(series, series_imp):
    imp_mean=series_imp.describe()[1]
    imp_stddev=series_imp.describe()[2]
    meanval=series.describe()[1]
    stddev=series.describe()[2]
    accur=(abs(imp_mean-meanval)/meanval)**2 + (abs(imp_stddev - stddev)/stddev)**2
    return np.sqrt(accur)

#check the different knn imputer properties, which one preserves the moments?
accur=pd.DataFrame(np.zeros((50, len(train.columns[3:-1]))), columns=train.columns[3:-1])
train_red=train[train.columns[3:-1]]
for k in range(1, 50):
    imputer=KNN(n_neighbors=k)
    train_imputed=imputer.fit_transform(train_red)
    train_imputed=pd.DataFrame(train_imputed, columns=train_red.columns)
    for col in train_red.columns:
        
        series=train_red[col]
        series_imputed=train_imputed[col]
        
        accur.loc[k, col]=imput_accur_moments(series, series_imputed)
        
#next: try iterateive imputer
#afterwards: check other metrices, such as removing some values, use the imputer, see if they get back

def imput_accur_remove(original, imputed, replaced):
    #all three are pd.Dataframes, in replaced a True always when the value was replaced
    accuracy={}
    for col in replaced.columns:
        rep=np.array(replaced[col][replaced[col]==True].index)
        org=original.loc[rep,col]
        imp=imputed.loc[rep,col]
        if org.dtype!=bool:
            accur=np.mean(abs(np.array(org)-np.array(rep))/np.array(org))
        else:
            accur=np.mean(abs(np.array(org)-np.array(rep)))
        accuracy[col]=accur
    return accuracy

###replace the columns in the train set by boolean values for such as sex, DifficultyInBreathing etc.
binarycolumns=['Sex','Cough','DifficultyInBreathing','RespiratoryFailure', 'CardiovascularDisease'], 
train_bin=pd.DataFrame([])
for col in train_red.columns:
    if col not in binarycolumns:
        train_bin[col]=train_red[col]
    else: 
        coef=[train_bin[col].unique()]
        for c in coef:
            if np.isnan(c)==True:
                coef.remove(c)
        ###now we have the two binary values in the column, which can be referred to True and False
        ###train.Cough[train.Cough==0]=2


        
        
    
    #remove 10% of the remaining values per columns, and compare afterwards
    
    
    


        






