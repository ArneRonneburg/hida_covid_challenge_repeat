# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 20:17:47 2021

@author: Martin
Little Data exploration with the covid data
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import missingno as msno

####pandas option#####
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

####daten input####
train=pd.read_csv("Data/trainSet.txt")

#lable encoder
le=LabelEncoder()

list=["Sex","Cough","Hospital","Prognosis"]

train["Sex"]=le.fit_transform(train["Sex"])
train["Cough"]=le.fit_transform(train["Cough"])
train["Hospital"]=le.fit_transform(train["Hospital"])
train["Prognosis"]=le.fit_transform(train["Prognosis"])


#print(df.head())

#print(df.columns.values)

#missing enteries  in %
t = 30 #trigger in % für markante werte
print("|Name|Fehlende daten|Prozent|Markant|")
print("|----|---------------|---------|–––––––|")# Für GitHub
for i in range(len(train.columns)): ### range(len())-> enumerate
    name=train.columns[i]
    data_fehlen=train[train.columns[i]].isna().sum()
    prozent=round(data_fehlen / len(train) * 100)
    markant=prozent>t
    print("|",name,"|",data_fehlen,"|",prozent,"|",markant,"|")

print("\n")
print("Extrahierte markante Werte")
for i in range(len(train.columns)):
    name=train.columns[i]
    data_fehlen=train[train.columns[i]].isna().sum()
    prozent=round(data_fehlen / len(train) * 100)
    if prozent>t:
        print("|",name,"|",data_fehlen,"|",prozent,"|")


#Anzahl an Patienten im KH
Hospital=train.groupby('Hospital')["PatientID"].nunique()

#Heatmap der daten
#plt.figure(figsize=(10, 6))
#sns.heatmap(df.isna(), cbar=False, cmap="plasma",yticklabels=False)
#plt.show()

train_sort_Hosp = train.sort_values(by="Hospital")
plt.figure(figsize=(10, 6))
sns.heatmap(train_sort_Hosp.isna(), cbar=False, cmap="plasma",yticklabels=False)
#plt.show()

msno.bar(train_sort_Hosp, figsize=(12, 6), fontsize=12, color='steelblue')

#  imagefile, patientid, enfernen (Weil nicht relevant)
train_sort_Hosp_use=train_sort_Hosp.drop(['PatientID',"ImageFile"], axis=1)
plt.figure(figsize=(10, 6))
sns.heatmap(train_sort_Hosp_use.isna(), cbar=False, cmap="plasma",yticklabels=False)
#plt.show()

##scatterplot###
plt.figure(figsize=(15, 6))
y=sns.scatterplot(data=train_sort_Hosp_use)
y.set(yscale="log")
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
#plt.show()

###Correlation
plt.figure(figsize=(10, 6))
correlations=train_sort_Hosp_use.corr()
sns.heatmap(correlations, cmap="inferno")
#plt.show()
