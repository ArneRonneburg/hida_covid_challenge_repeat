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
from xgboost import XGBClassifier as XGB


from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

def imputation(df,k):
    if k==0:
        imputer=IterativeImputer(imputation_order='random')
    else:
        imputer=KNN(n_neighbors=k)
    df_imputed=imputer.fit_transform(df)
    df_imputed=pd.DataFrame(df_imputed, columns=df.columns)
    return df_imputed

def clip_df(df):
        
    limitations={
            "Age": [1, 120],
            "Temp_C" : [36, 42.5], 
             "WBC" : [0, 1000], 
            "CRP": [0, 2500],
            # "Fibrinogen": [5,10000], 
            "LDH": [0, 25000],
            "Ddimer": [10, 1000000],
            "Ox_percentage": [25, 100],
            "PaO2": [0,700],
            "SaO2": [0,100],
            "pH": [5, 10]
            }
    for key in limitations.keys():
        limits=limitations[key]
        df[key].clip(limits[0], limits[1], inplace=True)
    return df

def classification(df,y_truth,clf):
    Xtrain, Xtest,ytrain ,ytest=tts(df, y_truth)
    clf.fit(Xtrain, ytrain)
    pred=clf.predict(Xtest).round()
    pred_acc=f1_score(ytest, pred)
    return pred_acc


def my_pipeline(df, k, clf, n_iterations):
    y=df['Prognosis']
    X=df.drop(columns=['Prognosis', 'ImageFile', 'Hospital'])
    result=0
    for i in range(n_iterations):
        X=imputation(X, k)
        X=clip_df(X)
    
        result=result + classification(X, y,clf)
    return result/n_iterations
    



PATH=r"C:\Users\Arne\Documents\gits\hida_covid_challenge_repeat/"
train_pre=pd.read_csv(PATH + "trainSet.txt", index_col=None)
test_pre=pd.read_csv(PATH + "testSet.txt", index_col=None)
image_train=pd.read_csv(PATH + "ImagePredictions_Train.txt", index_col=None)
image_test=pd.read_csv(PATH + "ImagePredictions_Test.txt", index_col=None)

names=image_train.iloc[:,0].to_numpy()
for i, n in enumerate(names):
    names[i]=n[2:-3]
image_train.iloc[:,0]=names

names=image_test.iloc[:,0].to_numpy()
for i, n in enumerate(names):
    names[i]=n[2:-3]
image_test.iloc[:,0]=names



train=train_pre.merge(image_train,left_on='PatientID', right_on="ID").drop(columns=['PatientID','ID'])
test=test_pre.merge(image_test,left_on='PatientID', right_on="ID", how='outer').drop(columns=['PatientID','ID'])

le=LabelEncoder()
label =["Sex","Cough","Hospital","Prognosis"]
for i in label:
    train[i]=le.fit_transform(train[i])
    # test[i]=le.transform(test[i])


RFC=ensemble.RandomForestClassifier
ETR=ensemble.ExtraTreesRegressor
BR=linear_model.BayesianRidge
GBR=ensemble.GradientBoostingRegressor
ARD=linear_model.ARDRegression
LogR=linear_model.LogisticRegression
Ridge=linear_model.RidgeClassifier

models=['RFC','ETR', 'BR','GBR','ARD',  'LogR', 'Ridge', 'XGB']
overall_accuracy=pd.DataFrame([], columns=models, index=np.arange(0, 51))
for k in range(0, 51):
    print(k)
    for model in models:
        clf=eval(model + "()")
        result=my_pipeline(train, k, clf, 10)
        overall_accuracy.loc[k,model]=result
clf = RFC()
k=19
y=train['Prognosis']
imputed_train=imputation(train.drop(columns=['Prognosis', 'ImageFile', 'Hospital']), 19)
clipped_train=clip_df(imputed_train)

X=clipped_train.drop(columns=['Fibrinogen'])
clf.fit(X,y)
#-> and the winner is (even with only the second highest accuracy): RFC and k=19   

le=LabelEncoder()
label =["Sex","Cough","Hospital","Prognosis"]
for i in label:
    test[i]=le.fit_transform(test[i])
test=test.drop(columns=['Prognosis', 'ImageFile', 'Hospital', 'Fibrinogen'])
# imputer=KNN(n_neighbors=k)
# df_imputed=imputer.fit_transform(test)
# df_imputed=pd.DataFrame(df_imputed, columns=test.columns)
imputed_test=imputation(test, 19)
clipped_test=clip_df(imputed_test)
X_test=clipped_test
predictions=clf.predict(X_test)

solution=pd.read_csv(r"C:\Users\Arne\Documents\gits\hida_covid_challenge_repeat\solution/solutionTestSet.txt")
solution=solution.loc[:,['ImageFile', 'Prognosis']]
ids=solution.iloc[:,0].to_numpy()
for i in range(0, len(ids)):
    ids[i]=ids[i][:-4]
solution.iloc[:,0]=ids
res=pd.DataFrame([])
res['ID']=test_pre['PatientID']
res['pred']=predictions

res=res.merge(solution, left_on='ID', right_on="ImageFile")
y_pred=res['pred'].to_numpy()
y_truth=res['Prognosis'].to_numpy()
for i in range(len(y_truth)):
    if y_truth[i]=='MILD':
        y_truth[i]=0
    else:
        y_truth[i]=1
accuracy= f1_score(np.array(y_truth, dtype=int), y_pred)