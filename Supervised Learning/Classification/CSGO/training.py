import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from lazypredict.Supervised import LazyClassifier



# Statistics + Data visualization
df = pd.read_csv('csgo_after_processing.csv')
# profile = ProfileReport(df, title="Profiling Report", explorative=True)
# profile.to_file('csgo.html')


target = "result"
x = df.drop(target, axis=1)
y = df[target]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

nom_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder())
])


preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num_transformer, ["wait_time_s", "ping", "kills", "assists", "deaths", "mvps", "hs_percent", "points"]),
    ("nom_feature", nom_transformer, ["map"]),
])

clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=7, n_estimators=100, criterion="gini")),
])

params = {
    "model__n_estimators": [100, 200, 300],
    "model__criterion": ["gini", "entropy", "log_loss"],
}

# grid = GridSearchCV(estimator=clf, param_grid=params, cv=4, scoring="precision", verbose=2, n_jobs=4)
# grid.fit(x_train, y_train)


# print(grid.best_estimator_)
# print(grid.best_params_)
# print(grid.best_score_)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))

