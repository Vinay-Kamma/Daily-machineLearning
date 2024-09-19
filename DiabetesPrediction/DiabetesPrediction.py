import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import  svm
from sklearn.metrics import accuracy_score
df=pd.read_csv('DiabetesPrediction/Dataset/diabetes.csv')
sc=StandardScaler()
input_variables=df.drop(columns=['Outcome'])
output_variables=df[['Outcome']]
input_variables=sc.fit_transform(input_variables)
X_train,X_test,Y_train,Y_test=train_test_split(input_variables,output_variables,test_size=0.2,stratify=output_variables,random_state=1)
model=svm.SVC(kernel='linear')
model.fit(X_train,Y_train)
print("train Accuracy:",accuracy_score(model.predict(X_train),Y_train))
print("test Accuracy:",accuracy_score(model.predict(X_test),Y_test))