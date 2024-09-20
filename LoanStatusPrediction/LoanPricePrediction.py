import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df=pd.read_csv('/workspaces/Daily-machineLearning/LoanStatusPrediction/Dataset/LoanDataset.csv')
df.fillna({'Loan_Amount_Term':360},inplace=True)
df.dropna(inplace=True)
cat_cols=['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
num_cols=['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
le=LabelEncoder()
st=StandardScaler()
def encode_or_normalise(data,labels,encoder):
    for i in labels:
        data[i]=encoder.fit_transform(data[i])
encode_or_normalise(df,cat_cols,le)
df[num_cols]=st.fit_transform(df[num_cols])
features=df.drop(columns=['Loan_ID','Loan_Status'])
target=df['Loan_Status']
x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.2,random_state=2)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy Score:",accuracy_score(y_pred,y_test))
