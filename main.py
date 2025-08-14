import pandas as pd

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train_index = ~train.duplicated()
train = train.loc[train_index]
y_train = train['SalePrice']
X_train = train.drop('SalePrice', axis=1)

fill_values = {}

for col in X_train.columns:
    if X_train[col].dtype == 'object':
        fill_values[col] = X_train[col].mode()[0]
    else:
        fill_values[col] = X_train[col].median()

X_train = X_train.fillna(fill_values)
X_test = test.fillna(fill_values)

