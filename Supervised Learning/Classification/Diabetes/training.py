import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from lazypredict.Supervised import LazyClassifier

# Statistics + Data visualization
df = pd.read_csv('diabetes.csv')
# profile = ProfileReport(df, title="Profiling Report", explorative=True)
# profile.to_file('diabetes.html')


target = "Outcome"
x = df.drop(target, axis=1)
y = df[target]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)

# Data Preprocessing
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train model
model = RandomForestClassifier(random_state=100)
# model.fit(x_train, y_train)

# y_pred = model.predict(x_test)

# Model evaluation
# print(classification_report(y_test, y_pred))

params = {
    "n_estimators": [100, 200, 300],
    "criterion": ["gini", "entropy", "log_loss"],
}


grid = GridSearchCV(estimator=model, param_grid=params, cv=4, scoring="recall", verbose=2)
grid.fit(x_train, y_train)
print(grid.best_estimator_)
print(grid.best_params_)
print(grid.best_score_)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))

clf = LazyClassifier(verbose=0, ignore_warnings=None, custom_metric=None)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)
print(models)


