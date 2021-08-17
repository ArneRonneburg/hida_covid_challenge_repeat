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
from sklearn import preprocessing
import random

encoder = preprocessing.LabelEncoder()
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
y=encoder.fit_transform(train.Prognosis)

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


###replace the columns in the train set by boolean values for such as sex, DifficultyInBreathing etc.
binarycolumns=['Sex','Cough','DifficultyInBreathing','RespiratoryFailure', 'CardiovascularDisease'] 
train_bin=train_red.copy()

#labelencoder


for col in binarycolumns:
    train_bin[col]=encoder.fit_transform(train_bin[col])
    train_bin[col]=train_bin[col].astype(bool)


def remove_random_values(df):
    df_removed=df.copy()
    ###the df where the data is removed
    df_index=df.copy()
    
    
    for col in df.columns:
        indices=list(df[col].dropna().index)
        set_nan=random.sample(indices, int(len(indices)/10))#remove roughly 10%
        df_removed.loc[np.array(set_nan), col] = np.nan
        df_index.loc[:,col]=False
        df_index.loc[np.array(set_nan), col]=True
    return df_removed, df_index

def imput_accuracy(original, imputed, indices):
    #all three are pd.Dataframes, 
    #in indices a True always when the value was replaced
    #maybe use other metrices for the differencce, e.g. the standard deviation?
    accuracy=pd.DataFrame([], columns=original.columns)
    for col in indices.columns:
        rep=np.array(indices[col][indices[col]==True].index)
        org=original.loc[rep,col]
        imp=imputed.loc[rep,col]
        if org.dtype!=bool:
            accur=np.mean(abs(np.array(org)-np.array(imp))/org.std())
        else:
            accur=np.mean(abs(np.array(org)-np.array(imp)))
            
        accuracy[col]=[accur]
    return accuracy
def do_the_imputation(df1, imputer):
    #introduce a rounding for binary columns
    #introduce a range for columns such as oxygen saturation or ph
    imputed=pd.DataFrame(imputer.fit_transform(df1), columns=df1.columns)
    binary_cols=['Age','Sex','Cough','DifficultyInBreathing','RespiratoryFailure', 'CardiovascularDisease'] 
    limitations={
        "Age": [1, 120],
        "Temp_C" : [36, 42.5], 
        "WBC" : [0, 1000], 
        "CRP": [0, 2500],
        "Fibrinogen": [5,10000], 
        "LDH": [0, 25000],
        "Ddimer": [10, 1000000],
        "Ox_percentage": [25, 100],
        "PaO2": [0,700],
        "SaO2": [0,100],
        "pH": [5, 10]
        }
        
    for i in binary_cols:
        imputed[i]=np.round(np.array(imputed.loc[:,i]))
    for L in limitations.keys():
        low, high= limitations[L]
        imputed[i]=np.clip(np.array(imputed.loc[:,i]),low, high)
    return imputed
imputer_accuracy=np.zeros(75)
for j in range(1, len(imputer_accuracy)):
    print(j)
    imputer=KNN(n_neighbors=j)
    
    df1,df2=remove_random_values(train_bin)
    imputed=do_the_imputation(df1,imputer)
    
    acc= imput_accuracy(train_bin, imputed, df2)
    for i in range(1, 10):
        df1,df2=remove_random_values(train_bin)
        imputed=pd.DataFrame(imputer.fit_transform(df1), columns=df1.columns)
        acc=acc + imput_accuracy(train_bin, imputed, df2)
    imputer_accuracy[j]=acc.mean(axis=1)

imputer=KNN(n_neighbors=15)
    

# imputed=pd.DataFrame(imputer.fit_transform(train_bin), columns=train_bin.columns)
# from sklearn.model_selection import train_test_split as tts    
# from sklearn.ensemble import ExtraTreesRegressor as ETR
# from sklearn.ensemble import RandomForestRegressor as RFR
# from sklearn.ensemble import RandomForestClassifier as RFC
# from sklearn.tree import DecisionTreeRegressor as DTR
# from sklearn.ensemble import AdaBoostRegressor as ABR
# from sklearn.ensemble import BaggingRegressor as BR
# from sklearn.ensemble import GradientBoostingRegressor as GBR
# from sklearn.ensemble import StackingRegressor as STR
# from sklearn.ensemble import VotingRegressor as VR
# from sklearn.experimental import enable_hist_gradient_boosting
# from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
# from sklearn.linear_model import BayesianRidge as BR
# from sklearn.linear_model import ARDRegression as ARD
# from xgboost import XGBRegressor as XGB
# from sklearn.linear_model import ElasticNet as EN
# from sklearn.linear_model import Lars
# from sklearn.linear_model import ElasticNetCV as ENCV
# from sklearn.linear_model import LogisticRegression as LogR
# from sklearn.linear_model import RidgeClassifier as Ridge
# from sklearn.linear_model import Perceptron, LinearRegression
# from sklearn.svm import LinearSVR as LinSVR
# Xtrain, Xtest,ytrain ,ytest=tts(imputed, y)
# RESULT={}
# models=['RFR','RFC','ETR', 'DTR', 'ABR','BR','GBR','XGB',
#             'HGBR', 'BR', 'ARD', 'EN', 'ENCV', 'Lars', 'LogR', 
#             'Ridge', 'Perceptron', 'LinearRegression', 'LinSVR']
# for name in models:

#     regressor=eval(name + "()")
    
#     regressor.fit(Xtrain, ytrain)
#     res=regressor.predict(Xtest).round()
#     pred_acc=1-np.mean(abs(res-ytest))
#     RESULT[name]=pred_acc

# regressors={}
# regressors['RFR']=RFR()
# regressors['BR']=BR()
# regressors['logR']=LogR()
# regressors['GBR']=GBR()
# import random
# result=[]
# models=['RFR','ETR', 'DTR', 'ABR','GBR','XGB',
#             'HGBR', 'BR', 'ARD', 'EN', 'ENCV', 'Lars', 
#             'LinearRegression', 'LinSVR']
# for i in range(0, 100):
#     print(i)
#     n_regressors=np.round(random.random()*9+3) #the number of employed regressors fro STR
    
    
#     estimators=[]
#     random_regressors=random.sample(list(np.arange(0, len(models))), int(n_regressors))
#     #get a list of indices, referring to the loaded regressors. Now stack them together
#     for i in random_regressors:
#         estimators+=[(models[i], eval(models[i]+"()"))]
#     regressor=STR(estimators=estimators)
#     regressor.fit(Xtrain, ytrain)
    
#     res=regressor.predict(Xtest).round()
#     pred_acc=1-np.mean(abs(res-ytest))
#     result.append(pred_acc)
# # ###same for VR
# # and maybe a random selection of regressors?
# #what about keras/tensorflow?
# ##reworking impuiatition, clipping to a certain range...
    
            

    


        
        
    
    
    
    
    
















        







