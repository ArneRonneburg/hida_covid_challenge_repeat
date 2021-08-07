import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

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

#print(df.head())

#print(df.columns.values)

#missing enteries  in %
for i in range(len(df.columns)):
    name = df.columns[i]
    data_fehlen = df[df.columns[i]].isna().sum()
    prozent = data_fehlen / len(df) * 100
    print('|%s|  Fehlende Daten: %d | Prozent %.2f| |' % (name, data_fehlen , prozent))


Hospital = df.groupby('Hospital')["PatientID"].nunique()
#print(Hospital)


#Heatmap der daten
plt.figure(figsize=(10, 6))
sns.heatmap(df.isna(), cbar=False, cmap="plasma",yticklabels=False)
#plt.show()

df_sort_Hosp = df.sort_values(by="Hospital")
plt.figure(figsize=(10, 6))
sns.heatmap(df_sort_Hosp.isna(), cbar=False, cmap="plasma",yticklabels=False)
#plt.show()

#  imagefile, patientid, enfernen (Weil nicht relevant)
df_sort_Hosp_use=df_sort_Hosp .drop(['PatientID',"ImageFile"], axis=1)
plt.figure(figsize=(10, 6))
sns.heatmap(df_sort_Hosp_use.isna(), cbar=False, cmap="plasma",yticklabels=False)
#plt.show()

###Correlation
correlations = df_sort_Hosp_use.corr()
sns.heatmap(correlations, cmap="inferno")
#plt.show()

