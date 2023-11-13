#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv(r"C:\Users\LENOVO\Downloads\car.csv")
print(data)

data.describe()

data.info()

# Calculate the percentage of each brand
brand_percentage = data['brand'].value_counts() / len(data) * 100

# Filter brands that make up 90% of the total cars
selected_brands = brand_percentage[brand_percentage.cumsum() <= 90].index

# Filter the DataFrame based on selected brands
data_filtered = data[data['brand'].isin(selected_brands)]


data_filtered.columns


data_filtered.isnull().sum()

data_filtered['brand'].unique()

data_filtered['brand'].value_counts()


data_filtered['brand'].value_counts().sum()


df= data_filtered.drop(['list_id', 'list_time', 'fuel','condition'], axis='columns')
print(df)


columns_fill_mode =['origin','type','gearbox','color']

for column in columns_fill_mode:
    mode_value = df[column].mode()[0]  # Calculate the mode
    df[column] = df[column].fillna(mode_value)  # Fill missing values with the mode


columns_fill_mean = ['seats', 'price']  # Replace with your numerical column names

for column in columns_fill_mean:
    mean_value = df[column].mean()  # Calculate the mean
    df[column] = df[column].fillna(mean_value)  # Fill missing values with the mean

df.isnull().sum()

'''
# In[14]:


df['log_price'] = np.log1p(df['price'])
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))
sns.distplot(df.log_price)
plt.show()


# In[15]:


plt.figure(figsize=(20, 15))
correlations = df.corr()
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()
'''

obdata = df.select_dtypes(include=object)
numdata = df.select_dtypes(exclude=object)

from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
for i in range(0,obdata.shape[1]):
    obdata.iloc[:,i] = lab.fit_transform(obdata.iloc[:,i])  

X= pd.concat([obdata,numdata],axis=1)

print(X)
print(df)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

X_train, X_test, y_train, y_test = train_test_split(X, df['price'], test_size=0.2, random_state=42)

algorithm = ['LinearRegression','DecisionTreeClassifier','RandomForestClassifier',
             'GradientBoostingRegressor','CatBoostingRegressor']
R2=[]
RMSE = []


from sklearn.metrics import r2_score, mean_squared_error


def models(model):
    model.fit(X_train,y_train)
    pre = model.predict(X_test)
    r2 = r2_score(y_test,pre)
    rmse = np.sqrt(mean_squared_error(y_test,pre))
    R2.append(r2)
    RMSE.append(rmse)
    score = model.score(X_test,y_test)
    print(f'The Score of Model is :{score}')


from sklearn.ensemble import GradientBoostingRegressor
model1 = LinearRegression()
model2 = DecisionTreeRegressor()
model3 = RandomForestRegressor()
model4 = GradientBoostingRegressor()
model5= CatBoostRegressor()


models(model1)
models(model2)
models(model3)
models(model4)
models(model5)

df1 = pd.DataFrame({'Algorithm':algorithm, 'R2_score': R2, 'RMSE':RMSE})
df1

fig = plt.figure(figsize=(20,8))
plt.plot(df1.Algorithm,df1.R2_score ,label='R2_score',lw=5,color='peru',marker='v',markersize = 15)
plt.legend(fontsize=15)
plt.show()


fig = plt.figure(figsize=(20,8))
plt.plot(df1.Algorithm,df1.RMSE ,label='RMSE',lw=5,color='r',marker='o',markersize = 10)
plt.legend(fontsize=15)
plt.show()

import pickle
import joblib
pickle.dump(model3, open('../model_HR.pkl', 'wb'))
joblib.dump(model3, open('../model_HR.sav', 'wb'))

import json
brand={'brand_value':list(X.brand.unique())}
with open ('../brand_value.json', 'w') as f:
    f.write(json.dumps(brand))
origin={'origin_value':list(X.origin.unique())}
with open ('../origin_value.json', 'w') as f:
    f.write(json.dumps(origin))
type={'type_value':list(X.type.unique())}
with open ('../type_value.json', 'w') as f:
    f.write(json.dumps(type))
gearbox={'gearbox_value':list(X.gearbox.unique())}
with open ('../gearbox_value.json', 'w') as f:
    f.write(json.dumps(gearbox))

