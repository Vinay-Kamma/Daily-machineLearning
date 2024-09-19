import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot  as plt
from xgboost import XGBRegressor
import seaborn as sns
from sklearn.metrics import mean_squared_error,r2_score
df=pd.read_csv('/workspaces/Daily-machineLearning/HousePricePrediction/Dataset/BostonHousing.csv')
print(df.head())
print(df.isnull().sum())
features=df.drop('medv',axis=1)
target=df['medv']
co_relation=df.corr()
print(co_relation)
x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.2,random_state=2)
regressor=XGBRegressor()
regressor.fit(x_train,y_train)
Y_pred=regressor.predict(x_test)
print('mean square error:',mean_squared_error(Y_pred,y_test))
print('r2 score:',r2_score(Y_pred,y_test))
# print(regressor.predict([x_test.loc[3]],y_test[3]))





