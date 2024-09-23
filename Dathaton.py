import pandas as pd
#import matplotlib.pyplot as plt
#import missingno as msno
#from scipy import stats
#import math
#from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
#import seaborn as sns
import numpy as np
import xgboost as xgb
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder

#Importing data
train=pd.read_csv('C:/Users/99559/OneDrive - iset.ge/Desktop/ტყემჩო/Dathaton/train.csv')
test=pd.read_csv('C:/Users/99559/OneDrive - iset.ge/Desktop/ტყემჩო/Dathaton/test.csv')

#Setting patient_id as an index
train.set_index('patient_id', inplace=True)
test.set_index('patient_id', inplace=True)

#Dropping column with only one value
train.drop(columns='patient_gender', inplace=True)
test.drop(columns='patient_gender', inplace=True)

target='metastatic_diagnosis_period'

#Handling missing values-------------------------------------------------------

#Dropping variables with more than 50% missing values
for column in train:
    if (len(train)-train[column].count())/len(train) >= 0.5:
            train.drop(columns=column, inplace=True)
            test.drop(columns=column, inplace=True)

#Filling numerical values using KNN
numerical_features_train = train.select_dtypes(include=['float64', 'int64']).columns
imputer = KNNImputer(n_neighbors=5)
train[numerical_features_train] = imputer.fit_transform(train[numerical_features_train])

numerical_features_test = test.select_dtypes(include=['float64', 'int64']).columns
imputer = KNNImputer(n_neighbors=5)
test[numerical_features_test] = imputer.fit_transform(test[numerical_features_test])

#Filling categorical values with mode
categorical_features_train = train.select_dtypes(include=['object', 'category']).columns
mode_imputer = SimpleImputer(strategy='most_frequent')
train[categorical_features_train] = mode_imputer.fit_transform(train[categorical_features_train])

categorical_features_test = test.select_dtypes(include=['object', 'category']).columns
mode_imputer = SimpleImputer(strategy='most_frequent')
test[categorical_features_test] = mode_imputer.fit_transform(test[categorical_features_test])

#Target Encoding---------------------------------------------------------------------- 
categorical_train=[]
for column in train:
    if train[column].dtype==object:
        train[column]=train[column].astype(str)
        categorical_train.append(column)
        encoder = TargetEncoder(cols=column)
        train[column] = encoder.fit_transform(train[column], train[target])
        test[column] = encoder.transform(test[column])
        
#Choosing variables------------------------------------------------------------

X_train=train.drop(columns=target, inplace=False)
y_train=train[target]

corr_target = dict(X_train.corrwith(y_train).abs().sort_values(ascending=False))

keep_corr_target=[]
for value in corr_target:
    if corr_target[value] >= 0.01:
        keep_corr_target.append(value)
    else:
        X_train.drop(columns=value, inplace=True)

corr_matrix = X_train.corr().abs()
mask = corr_matrix.applymap(lambda x: True if x > 0.8 else False)
highly_correlated_pairs = [(i, j) for i in mask.columns for j in mask.columns if mask.loc[i, j]]

highly_correlated_pairs = []
for i in range(len(mask.columns)):
    for j in range(i + 1, len(mask.columns)):  # Start from i+1 to avoid duplicates
        if mask.iloc[i, j]:
            highly_correlated_pairs.append((mask.columns[i], mask.columns[j]))

features_to_drop=[]
for feature1, feature2 in highly_correlated_pairs:
    if feature2 not in features_to_drop:
        X_train.drop(columns=feature2, inplace=True)
        features_to_drop.append(feature2)
        
#Fitting the model-------------------------------------------------------------

# Split the data into training and validation sets
X_train_spl, X_val, y_train_spl, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}

# Initialize the Gradient Boosting regressor
gbr = GradientBoostingRegressor()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_root_mean_squared_error')

# Fit GridSearchCV
grid_search.fit(X_train_spl, y_train_spl)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the model on the validation set
y_val_pred = best_model.predict(X_val)
validation_rmse = mean_squared_error(y_val, y_val_pred, squared=False)

print("Best parameters found: ", best_params)
print("Validation RMSE: ", validation_rmse)
#Performing random forest and dropping variables with small feature importance
'''RANDOM FOREST'''
'''
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
importances=pd.DataFrame(importances)
importances.set_index(X_train.columns, inplace=True)

for column in importances.index:
    if importances.at[column, 0]<=0.001:
        importances.drop(index=column, inplace=True)
        X_train.drop(columns=column, inplace=True)
        '''
   
'''XGBOOST'''
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, learning_rate=0.3, subsample=0.8)
xg_reg.fit(X_train, y_train)


gb_model = GradientBoostingRegressor(learning_rate = 0.05, max_depth = 3, min_samples_leaf = 4, 
                                      min_samples_split = 2, n_estimators = 100, subsample = 1.0)
gb_model.fit(X_train, y_train)

#------------------------------------------------------------------------------        
X_test=test.copy()

for column in X_test:
    if column not in X_train:
        X_test.drop(columns=column, inplace=True)
    
y_train=pd.DataFrame(y_train)
y_train['Predicted']=gb_model.predict(X_train)

y_test=pd.DataFrame(columns=['patient_id','metastatic_diagnosis_period'])
y_test['patient_id']=X_test.index
y_test['metastatic_diagnosis_period']=gb_model.predict(X_test)

y_test.to_csv('submission.csv', index=False)

rmse = np.sqrt(np.mean((y_train['metastatic_diagnosis_period'] - y_train['Predicted']) ** 2))

param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.05],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the model
gb = GradientBoostingRegressor(random_state=42)

# Initialize GridSearchCV
grid_search_gb = GridSearchCV(estimator=gb, param_grid=param_grid_gb, 
                              cv=3, n_jobs=-1, scoring='neg_mean_squared_error')

# Fit GridSearchCV
grid_search_gb.fit(X_train, y_train)

# Best parameters and model
best_gb = grid_search_gb.best_estimator_
print("Best parameters for Gradient Boosting:", grid_search_gb.best_params_)

#Hyperparameter tuning---------------------------------------------------------
#RandomForest parameter choosing 
'''
param_grid = { 
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_train)
rmse_rf = mean_squared_error(y_train, y_pred, squared=False)
print(f'Best RF RMSE: {rmse_rf}')
'''

#XGBoost parameter choosing
''' 
param_grid = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 0.9, 1.0]
}

xg_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=100)

grid_search = GridSearchCV(estimator=xg_reg, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_xgboost = grid_search.best_estimator_
y_pred = best_xgboost.predict(X_train)
rmse_xgboost = mean_squared_error(y_train, y_pred, squared=False)
print(f'Best XGBoost RMSE: {rmse_xgboost}')
'''