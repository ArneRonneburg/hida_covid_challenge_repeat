import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split,cross_val_score,RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
#from sklearn.svm import SVR
from sklearn.metrics import classification_report, confusion_matrix

####pandas option#####
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

####daten input####
df= pd.read_csv("Data/trainSet.txt")

#lable encoder
le = LabelEncoder()
df["Sex"]=le.fit_transform(df["Sex"])
df["Cough"]=le.fit_transform(df["Cough"])
df["Hospital"]=le.fit_transform(df["Hospital"])
df["Prognosis"]=le.fit_transform(df["Prognosis"])


#missing enteries  in %
for i in range(len(df.columns)):
    data_fehlen = df[df.columns[i]].isna().sum()
    prozent = data_fehlen / len(df) * 100
    print('>%d,  Fehlende Daten: %d, Prozent %.2f' % (i, data_fehlen , prozent))

#Heatmap der daten
plt.figure(figsize=(10, 6))
sns.heatmap(df.isna(), cbar=False, cmap="plasma",yticklabels=False)
#plt.show()


#  imagefile, patientid, enfernen (Weil nicht relevant)
df_use=df.drop(['PatientID',"ImageFile"], axis=1)
plt.figure(figsize=(10, 6))
sns.heatmap(df_use.isna(), cbar=False, cmap="plasma",yticklabels=False)
#plt.show()
