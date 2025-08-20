import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train_index = ~train.duplicated()
train = train.loc[train_index]
y_train = train['SalePrice']
X_train = train.drop('SalePrice', axis=1)

numeric_cols = X_train.select_dtypes(include='number').columns
categorical_cols = X_train.select_dtypes(include='object').columns

for col in numeric_cols:
    lower_bound = X_train[col].quantile(0.01)
    upper_bound = X_train[col].quantile(0.99)
    X_train[col] = X_train[col].clip(lower=lower_bound, upper=upper_bound)
    test[col] = test[col].clip(lower=lower_bound, upper=upper_bound)

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
    ('classifier', XGBRegressor())
])

pipeline.fit(X_train, y_train)

