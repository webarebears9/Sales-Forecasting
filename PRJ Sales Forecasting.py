#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6


# In[ ]:


os.chdir ("C:\\Users\\anilb\\OneDrive\\Desktop\\clg\\acmegrade assignments")


# In[ ]:


display (os.getcwd())


# In[ ]:


dt = pd.read_csv('Train.csv')
display (dt.head())


# In[ ]:


print (dt.shape)


# In[ ]:


display (dt.columns)


# In[ ]:


display (dt.describe())


# In[ ]:


display (dt.info())


# In[ ]:


display (dt.apply(lambda x: len(x.unique())))


# In[ ]:


display (dt.isnull().sum())


# In[ ]:


cat_col = []
for x in dt.dtypes.index:
    if dt.dtypes[x] == 'object':
        cat_col.append(x)
display (cat_col)


# In[ ]:


cat_col.remove('Item_Identifier')
cat_col.remove('Outlet_Identifier')
display (cat_col)


# In[ ]:


for col in cat_col:
    print(col , len(dt[col].unique()))


# In[ ]:


for col in cat_col:
    print(col)
    print(dt[col].value_counts())
    print()
    print ('*' *50)


# In[ ]:


miss_bool = dt['Item_Weight'].isnull()
display (miss_bool)


# In[ ]:


display (dt['Item_Weight'].isnull().sum())


# In[ ]:


Item_Weight_null = dt[dt['Item_Weight'].isna()]
display (Item_Weight_null)


# In[ ]:


Item_Weight_null['Item_Identifier'].value_counts()


# In[ ]:


item_weight_mean = dt.pivot_table(values = "Item_Weight", index = 'Item_Identifier')
display (item_weight_mean)


# In[ ]:


display (dt['Item_Identifier'])


# In[ ]:


for i, item in enumerate(dt['Item_Identifier']):
    if miss_bool[i]:
        if item in item_weight_mean.index:
            dt['Item_Weight'][i] = item_weight_mean.loc[item]['Item_Weight']
        else:
            dt['Item_Weight'][i] = np.mean(dt['Item_Weight'])


# In[ ]:


result = dt['Item_Weight'].isnull().sum()
display (result)


# In[ ]:


result = dt.groupby('Outlet_Size').agg({'Outlet_Size': np.size})
display (result)


# In[ ]:


result= dt['Outlet_Size'].isnull().sum()
display (result)


# In[ ]:


Outlet_Size_null= dt[dt['Outlet_Size'].isna()]
display (Outlet_Size_null)


# In[ ]:


result = Outlet_Size_null['Outlet_Type'].value_counts()
display (result)


# In[ ]:


result= dt.groupby (['Outlet_Type','Outlet_Size'] ).agg({'Outlet_Type':[np.size]})
display (result)


# In[ ]:


outlet_size_mode = dt.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
display (outlet_size_mode)


# In[ ]:


miss_bool = dt['Outlet_Size'].isnull()
dt.loc[miss_bool, 'Outlet_Size'] = dt.loc[miss_bool, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
miss_bool = dt['Outlet_Size'].isnull()
for i, item in enumerate (dt['Outlet_Size']):
    if miss_bool[i]:
        dt['Outlet_Size'][i] = outlet_size_mode.loc['Outlet_Size',dt['Outlet_Type'][i] ]


# In[ ]:


display (dt['Outlet_Size'].isnull().sum())


# In[ ]:


result = dt.groupby (['Outlet_Type','Outlet_Size'] ).agg({'Outlet_Type':[np.size]})
display (result)


# In[ ]:


display (sum(dt['Item_Visibility']==0))


# In[ ]:


dt.loc[:, 'Item_Visibility'].replace([0], [dt['Item_Visibility'].mean()], inplace=True)


# In[ ]:


display(sum(dt['Item_Visibility']==0))


# In[ ]:


result = dt['Item_Fat_Content'].value_counts()
display (result)


# In[ ]:


dt['Item_Fat_Content'] = dt['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
result = dt['Item_Fat_Content'].value_counts()
display (result)


# In[ ]:


dt['New_Item_Type'] = dt['Item_Identifier'].apply(lambda x: x[:2])
display (dt['New_Item_Type'])


# In[ ]:


display (dt['New_Item_Type'].value_counts())


# In[ ]:


dt['New_Item_Type'] = dt['New_Item_Type'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})
display (dt['New_Item_Type'].value_counts())


# In[ ]:


display (dt['Item_Fat_Content'].value_counts())


# In[ ]:


result = dt.groupby (['New_Item_Type','Item_Fat_Content'] ).agg({'Outlet_Type':[np.size]})
display (result)


# In[ ]:


dt.loc[dt['New_Item_Type']=='Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
result =  (dt['Item_Fat_Content'].value_counts())
display (result)



# In[ ]:


result = dt.groupby (['New_Item_Type','Item_Fat_Content'] ).agg({'Outlet_Type':[np.size]})
display (result)


# In[ ]:


dt['Outlet_Years'] = 2024 - dt['Outlet_Establishment_Year']
print (dt['Outlet_Years'])


# In[ ]:


display (dt.head())


# In[ ]:


sns.distplot(dt['Item_Weight'])
plt.show()


# In[ ]:


sns.distplot(dt['Item_Visibility'])
plt.show()


# In[ ]:


sns.distplot(dt['Item_MRP'])
plt.show()


# In[ ]:


sns.distplot(dt['Item_Outlet_Sales'])
plt.show()


# In[ ]:


dt['Item_Outlet_Sales'] = np.log(1+dt['Item_Outlet_Sales'])
display (dt['Item_Outlet_Sales'])


# In[ ]:


sns.distplot(dt['Item_Outlet_Sales'])
plt.show()


# In[ ]:


sns.countplot(x = dt["Item_Fat_Content"])
plt.show()


# In[ ]:


l = list(dt['Item_Type'].unique()) 
chart = sns.countplot(x =dt["Item_Type"])
chart.set_xticklabels(labels=l, rotation=90)
plt.show()



# In[ ]:


sns.countplot(x= dt['Outlet_Establishment_Year'])
plt.show()


# In[ ]:


sns.countplot(x=dt['Outlet_Size'])
plt.show()


# In[ ]:


sns.countplot(x=dt['Outlet_Location_Type'])
plt.show()


# In[ ]:


sns.countplot(x= dt['Outlet_Type'])
plt.show()


# In[ ]:


display(dt.head(3))


# In[ ]:


dtc= dt.iloc[:,[1,3,5,7,11,13]]
display (dtc)


# In[ ]:


corr = dtc.corr()
display (corr)


# In[ ]:


sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()


# In[ ]:


display (dt.head())


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dt['Outlet'] = le.fit_transform(dt['Outlet_Identifier'])
display (dt['Outlet'])


# In[ ]:


cat_col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type']
for col in cat_col:
    dt[col] = le.fit_transform(dt[col])
display (dt.head())   


# In[ ]:


dt = pd.get_dummies(dt, columns=['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type'],dtype = int )
display (dt.head())


# In[ ]:


X = dt.drop(columns=['Outlet_Establishment_Year', 'Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
display (X.head())


# In[ ]:


y = dt['Item_Outlet_Sales']
display (y.head())


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print (X.shape, y.shape)
print (X_train.shape, X_test.shape ,  y_train.shape, y_test.shape)


# In[ ]:


from sklearn import metrics 
display (",   ".join(metrics.get_scorer_names()))


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
def train(model, X, y):
    # training the model
    model.fit(X, y)
    
    pred = model.predict(X)
    # perform cross-validation
    cv_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    print("Model Report")
    print ('Scoring - neg_mean_squared_error')
    print ( cv_score )    
    cv_score = np.abs(np.mean(cv_score))    
    print ('ABS Average of - neg_mean_squared_error',cv_score )       
    cv_score = cross_val_score(model, X, y,  cv=5)
    print ()
    print ('R2 Score ')
    print ( cv_score )    
    cv_score = np.mean(cv_score)     
    print ('Average R2 Score ',cv_score)    
    print ()
    print ('Accuracy for full Data')
    print('R2_Score:', r2_score(y,pred))
    print ()


# In[ ]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
model = LinearRegression()
train(model, X_train, y_train)
coef = pd.Series(model.coef_, X.columns).sort_values()
print (coef)
coef.plot(kind='bar', title="Model Coefficients")
plt.show()


# In[ ]:


model = Ridge()
train(model, X_train, y_train)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title="Model Coefficients")
plt.show()


# In[ ]:


model = Lasso()
train(model, X_train, y_train)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title="Model Coefficients")
plt.show()


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
train(model,X_train, y_train)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="Feature Importance")
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
train(model, X_train, y_train)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="Feature Importance")
plt.show()


# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
train(model, X_train, y_train)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="Feature Importance")
plt.show()


# In[ ]:


from lightgbm import LGBMRegressor
model = LGBMRegressor()
train(model, X_train, y_train)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="Feature Importance")
plt.show()


# In[ ]:


from xgboost import XGBRegressor
model = XGBRegressor()
train(model, X_train, y_train)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="Feature Importance")
plt.show()


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]


# In[ ]:


random_grid = {
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[ ]:


rf = RandomForestRegressor()
rf=RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = -1)
display (rf.fit(X_train, y_train))


# In[ ]:


print(rf.best_params_)
print(rf.best_score_)
predictions=rf.predict(X_test)
display (r2_score (y_test,predictions))
display (predictions)    


# In[ ]:


sns.distplot(y_test-predictions)
plt.show()


# In[ ]:


from scipy.stats import uniform, randint
params = {
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), # default 0.1 
    "max_depth": randint(2, 6), # default 3
    "n_estimators": randint(100, 150), # default 100
    "subsample": uniform(0.6, 0.4)
}


# In[ ]:


lgb=LGBMRegressor()
lgb = RandomizedSearchCV(estimator = lgb, param_distributions = params,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
lgb.fit(X,y)


# In[ ]:


print(lgb.best_params_)
print(lgb.best_score_)
predictions=lgb.predict(X_test)
display (r2_score (y_test,predictions))
display (predictions)   


# In[ ]:


sns.distplot(y_test-predictions)
plt.show()


# In[ ]:


params = {
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), # default 0.1 
    "max_depth": randint(2, 6), # default 3
    "n_estimators": randint(100, 150), # default 100
    "subsample": uniform(0.6, 0.4)
}


# In[ ]:


xgb = RandomizedSearchCV(estimator = model, param_distributions = params,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
xgb.fit(X,y)



# In[ ]:


print(xgb.best_params_)
print(xgb.best_score_)
predictions=xgb.predict(X_test)
display (r2_score (y_test,predictions))
display (predictions)


# In[ ]:


sns.distplot(y_test-predictions)
plt.show()

