import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

path = ""
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")

train['dataset'] = "train"
test['dataset'] = "test"

train['dataset'] = "train"
test['dataset'] = "test"
data = pd.concat([train,test], axis = 0)
categorical = ['property_type','room_type','bed_type','cancellation_policy','city']
data = pd.get_dummies(data, columns = categorical)

numerics = ['uint8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
train_x = data[data.dataset == "train"] \
    .select_dtypes(include=numerics) \
    .drop("log_price", axis = 1) \
    .fillna(0) \
    .values

test_x = data[data.dataset == "test"] \
    .select_dtypes(include=numerics) \
    .drop("log_price", axis = 1) \
    .fillna(0) \
    .values
    
train_y = data[data.dataset == "train"].log_price.values

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(train_x[:,63:64])
# Train a Random Forest model with cross-validation

'''from sklearn.model_selection import KFold
cv_groups = KFold(n_splits=3)
regr = RandomForestRegressor(random_state = 0, n_estimators = 10)'''
