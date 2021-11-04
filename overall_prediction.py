# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 20:48:52 2021

@author: Arne
"""


import random 
import time
import numpy as np
import pandas as pd
####define measures for the imputation quality. Idea 1: they have a similar mean and variance
from sklearn.impute import KNNImputer as KNN
from sklearn.experimental import enable_iterative_imputer  # noqa

from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts 
from sklearn import ensemble
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.experimental import enable_hist_gradient_boosting
from xgboost import XGBRegressor as XGB
from sklearn.svm import LinearSVR as LinSVR
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neural_network import MLPClassifier as MLPC

ETR=ensemble.ExtraTreesRegressor
RFR=ensemble.RandomForestRegressor
RFC=ensemble.RandomForestClassifier
ABR=ensemble.AdaBoostRegressor
BR=ensemble.BaggingRegressor
GBR=ensemble.GradientBoostingRegressor
STR=ensemble.StackingRegressor
VR=ensemble.VotingRegressor


BR=linear_model.BayesianRidge
ARD=linear_model.ARDRegression
EN=linear_model.ElasticNet
ENCV=linear_model.ElasticNetCV
LogR=linear_model.LogisticRegression
Ridge=linear_model.RidgeClassifier
Perceptron=linear_model.Perceptron
LR=linear_model.LinearRegression
Lars=linear_model.Lars


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


PATH=r"C:\Users\Arne\Documents\gits\hida_covid_challenge_repeat/"
train=pd.read_csv(PATH + "trainSet.txt", index_col=None)
test=pd.read_csv(PATH + "testSet.txt", index_col=None)

le=LabelEncoder()
label =["Sex","Cough","Hospital","Prognosis"]
for i in label:
    train[i]=le.fit_transform(train[i])

imputer=KNN(n_neighbors=30)
train_imputed=imputer.fit_transform(train.iloc[:,2:])
imputed=pd.DataFrame([], columns=train.columns, index=train.index)
imputed.iloc[:,0]=train.iloc[:,0]
imputed.iloc[:,1]=train.iloc[:,1]
for i in range(2, len(imputed.columns)):
    imputed.iloc[:,i]=train_imputed[:,i-2]
    
imageFiles=pd.read_csv(PATH + "ImagePredictions_Train.txt", index_col=None)
names=imageFiles.iloc[:,0].to_numpy()
for i, n in enumerate(names):
    names[i]=n[2:-3]
imageFiles.iloc[:,0]=names

imputed_and_imagepred=imputed.merge(imageFiles,left_on='PatientID', right_on="ID")
imputed_and_imagepred.drop(columns=['ID'], inplace=True)
y=imputed_and_imagepred['Prognosis']
X=imputed_and_imagepred.drop(columns=['Prognosis', 'PatientID', 'ImageFile', 'Hospital'])
Xtrain, Xtest,ytrain ,ytest=tts(X, y)


ETR=ensemble.ExtraTreesRegressor
RFR=ensemble.RandomForestRegressor
RFC=ensemble.RandomForestClassifier
ABR=ensemble.AdaBoostRegressor
BR=ensemble.BaggingRegressor
GBR=ensemble.GradientBoostingRegressor
STR=ensemble.StackingRegressor
VR=ensemble.VotingRegressor


BR=linear_model.BayesianRidge
ARD=linear_model.ARDRegression
EN=linear_model.ElasticNet
ENCV=linear_model.ElasticNetCV
LogR=linear_model.LogisticRegression
Ridge=linear_model.RidgeClassifier
Perceptron=linear_model.Perceptron
LR=linear_model.LinearRegression
Lars=linear_model.Lars




RESULT={}
models=['RFR','RFC','ETR', 'ABR','BR','GBR',#"HGBR", 'XGB', 'DTR'
              'BR', 'ARD', 'EN', 'ENCV','Lars', 'LogR',
            'Ridge', 'Perceptron', 'LR', 'LinSVR']
for name in models:

    regressor=eval(name + "()")
    
    regressor.fit(Xtrain, ytrain)
    res=regressor.predict(Xtest).round()
    pred_acc=1-np.mean(abs(res-ytest))
    RESULT[name]=pred_acc
    
for i in range(1, 20):
    Xtrain, Xtest,ytrain ,ytest=tts(X, y)

    
    ETR=ensemble.ExtraTreesRegressor
    RFR=ensemble.RandomForestRegressor
    RFC=ensemble.RandomForestClassifier
    ABR=ensemble.AdaBoostRegressor
    BR=ensemble.BaggingRegressor
    GBR=ensemble.GradientBoostingRegressor
    STR=ensemble.StackingRegressor
    VR=ensemble.VotingRegressor
    
    
    BR=linear_model.BayesianRidge
    ARD=linear_model.ARDRegression
    EN=linear_model.ElasticNet
    ENCV=linear_model.ElasticNetCV
    LogR=linear_model.LogisticRegression
    Ridge=linear_model.RidgeClassifier
    Perceptron=linear_model.Perceptron
    LR=linear_model.LinearRegression
    Lars=linear_model.Lars
   




    models=['RFR','RFC','ETR', 'ABR','BR','GBR',#"HGBR", 'XGB', 'DTR'
              'BR', 'ARD', 'EN', 'ENCV','Lars', 'LogR',
            'Ridge', 'Perceptron', 'LR', 'LinSVR']
    for name in models:
    
        regressor=eval(name + "()")
        
        regressor.fit(Xtrain, ytrain)
        res=regressor.predict(Xtest).round()
        pred_acc=1-np.mean(abs(res-ytest))
        RESULT[name]=RESULT[name]+pred_acc

for m in models:
    RESULT[m]=RESULT[m]/20    


