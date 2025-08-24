import numpy as np
import os
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# Import train and test files
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# Preserve id values
test_ids = test['Id']
train_ids = train['Id']

# Drop duplicates
train_index = ~train.duplicated()
train = train.loc[train_index]

# Drop id columns, apply log transformation to sale prices
y_train = np.log1p(train['SalePrice'])
X_train = train.drop(['SalePrice', 'Id'], axis=1)
test = test.drop(['Id'], axis=1)

# Create variables for numeric and categorical columns
numeric_cols = X_train.select_dtypes(include='number').columns
categorical_cols = X_train.select_dtypes(include='object').columns

# Get rid of outliers at 1st and 99th percentiles
for col in numeric_cols:
    lower_bound = X_train[col].quantile(0.01)
    upper_bound = X_train[col].quantile(0.99)
    X_train[col] = X_train[col].clip(lower=lower_bound, upper=upper_bound)
    test[col] = test[col].clip(lower=lower_bound, upper=upper_bound)

# Feature Engineering - create new columns that are based on combining other fields
X_train['TotalBathrooms'] = X_train['FullBath'] + 0.5 * X_train['HalfBath']
test['TotalBathrooms'] = test['FullBath'] + 0.5 * test['HalfBath']
X_train['TotalSF'] = X_train['TotalBsmtSF'] + X_train['1stFlrSF'] + X_train['2ndFlrSF']
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

# Create preprocessing pipelines for both numerical and categorical data
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor())
])

# Create param_dist for hyperparameter testing
param_dist = {
    'regressor__n_estimators': randint(100, 500),
    'regressor__max_depth': randint(3, 20),
    'regressor__learning_rate': uniform(0.01, 0.5),
    'regressor__subsample': uniform(0.5, 0.5)
}

# Create model based on pipeline, parameter list, and cross validation folds
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

# Fit data
random_search.fit(X_train, y_train)

# Print best parameters and score
print("Best parameters:", random_search.best_params_)
print("Best RMSE:", -random_search.best_score_)

# Take the exponent of log transformed predictions to get normal prediction
y_pred_log = random_search.predict(test)
y_pred = np.expm1(y_pred_log)

# Create submission for Kaggle competition
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': y_pred
})

# Create the results directory if it doesn't exist
if not os.path.exists('../results'):
    os.makedirs('../results')

# Export submission
submission.to_csv('../results/submission.csv', index=False)