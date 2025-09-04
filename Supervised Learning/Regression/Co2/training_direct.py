import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score

def create_time_series_data(data, window=5, target_size=3):
    i = 1
    while i < window:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i += 1

    i = 0
    while i < target_size:
        data["target_{}".format(i+1)] = data["co2"].shift(-i-window)
        i += 1

    data.dropna(axis=0, inplace=True)
    return data


df = pd.read_csv('co2.csv')
df["time"] = pd.to_datetime(df["time"])
df["co2"] = df["co2"].interpolate()
df = create_time_series_data(df)



target_size=3
x = df.drop(["time"] + ["target_{}".format(i+1) for i in range(target_size)], axis=1)
y = df[["target_{}".format(i+1) for i in range(target_size)]]

train_ratio = 0.8
num_samples = len(x)

x_train = x[:int(num_samples * train_ratio)]
y_train = y[:int(num_samples * train_ratio)]
x_test = x[int(num_samples * train_ratio):]
y_test = y[int(num_samples * train_ratio):]

regs = []
for i in range(target_size):
    regs.append(LinearRegression())

for i, reg in enumerate(regs):
    reg.fit(x_train, y_train["target_{}".format(i+1)])

r2 = []
mae = []
mse = []

for i, reg in enumerate(regs):
    y_predict = reg.predict(x_test)
    mae.append(mean_absolute_error(y_test["target_{}".format(i+1)], y_predict)) # the smaller the better
    mse.append(mean_squared_error(y_test["target_{}".format(i+1)], y_predict)) # the smaller the better
    r2.append(r2_score(y_test["target_{}".format(i+1)], y_predict)) # good ~1, bad ~0

print("R2: {}".format(r2))
print("MAE: {}".format(mae))
print("MSE: {}".format(mse))
