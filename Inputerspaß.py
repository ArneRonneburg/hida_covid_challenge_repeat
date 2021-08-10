import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split,cross_val_score,RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
#from sklearn.svm import SVR
from sklearn.metrics import classification_report, confusion_matrix
import missingno as msno
from missingpy import MissForest


####https://www.kaggle.com/residentmario/simple-techniques-for-missing-data-imputation####

####pandas option#####
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

####daten input####
train = pd.read_csv('Data/trainSet.txt')
test = pd.read_csv('Data/testSet.txt')
#lable encoder
le = LabelEncoder()
list =["Sex","Cough","Hospital","Prognosis"]
for i in list:
    train[i]=le.fit_transform(train[i])

for i in list:
    test[i]=le.fit_transform(test[i])

####Patienten und image raus
train= train.drop(["PatientID","ImageFile"],1)

####simple imputer


#####missingo

#msno.bar(df, figsize=(12, 6), fontsize=12, color='steelblue')
#plt.show()


##### missingpy/missingForest#####

mFimputer = MissForest()
X= train
X_impute = mFimputer.fit_transform(X)

df_mf=pd.DataFrame(data=X_impute, columns=['Hospital', 'Age', 'Sex', 'Temp_C', 'Cough', 'DifficultyInBreathing',
       'WBC', 'CRP', 'Fibrinogen', 'LDH', 'Ddimer', 'Ox_percentage', 'PaO2',
       'SaO2', 'pH', 'CardiovascularDisease', 'RespiratoryFailure',
       'Prognosis'])

print(df_mf.loss
