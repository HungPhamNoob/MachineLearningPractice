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

def create_time_series_data(data, window=5):
    i = 1
    while i < window:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i += 1
    data["target"] = data["co2"].shift(-i)
    data.dropna(axis=0, inplace=True)
    return data


df = pd.read_csv('co2.csv')
df["time"] = pd.to_datetime(df["time"])
df["co2"] = df["co2"].interpolate()
df = create_time_series_data(df)

# fig, ax = plt.subplots()
# ax.plot(df["time"], df["co2"], label="CO2")
# ax.set_xlabel("Time")
# ax.set_ylabel("CO2")
# ax.legend()
# plt.show()


x = df.drop(["time", "target"], axis=1)
y = df["target"]

train_ratio = 0.8
num_samples = len(x)

x_train = x[:int(num_samples * train_ratio)]
y_train = y[:int(num_samples * train_ratio)]
x_test = x[int(num_samples * train_ratio):]
y_test = y[int(num_samples * train_ratio):]

# khong can scaler do data co pham vi giong nhau roi

reg = LinearRegression()

reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)

# print("MAE: {}".format(mean_absolute_error(y_test, y_predict))) # the smaller the better
# print("MSE: {}".format(mean_squared_error(y_test, y_predict))) # the smaller the better
# print("R2 Score: {}".format(r2_score(y_test, y_predict))) # good ~1, bad ~0


current_data = [380.5, 400, 407, 390, 390.5]

for i in range(10):
    print("Input is {}".format(current_data))
    prediction = reg.predict([current_data])[0]
    print("CO2 in week {} is {}".format(i+1, prediction))
    current_data = current_data[1:] + [prediction]

