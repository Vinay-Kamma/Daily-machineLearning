import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
df=pd.read_csv("/workspaces/Daily-machineLearning/Rock_OR_Mine/Dataset/sonar.csv",header=None)
df = df.sample(frac = 1)
input_variables=df.drop(columns=[60])
target_variables=df.loc[:,60]
le=LabelEncoder()
target_variables=le.fit_transform(target_variables)
X_Train,X_test,Y_train,Y_test=train_test_split(input_variables,target_variables,test_size=0.2,random_state=1)
lr=LogisticRegression()
lr.fit(X_Train,Y_train)
train_predictions=lr.predict(X_Train)
test_predictions=lr.predict(X_test)
print("Train Score",accuracy_score(Y_train,train_predictions))
print("Test Score",accuracy_score(Y_test,test_predictions))