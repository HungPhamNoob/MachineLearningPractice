import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

df = pd.read_csv('StudentScore.xls')
# profile = ProfileReport(df, title="Profiling Report", explorative=True)
# profile.to_file('StudentScore.html')


target = "writing score"
x = df.drop(target, axis=1)
y = df[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# Missing Value
# imputer = SimpleImputer(strategy='median')
# x.train[["math score", "reading score"]] = imputer.fit_transform(x.train[["math score", "reading score"]])

# Gauss Distribution
# scaler = StandardScaler()
# x.train[["math score", "reading score"]] = scaler.fit_transform(x.train[["math score", "reading score"]])

# "Numerical" Missing Value
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
# result = num_transformer.fit_transform(x_train[["math score", "reading score"]])


# "Nominal" Missing Value
nom_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', OneHotEncoder())
])
# result = nom_transformer.fit_transform(x_train[["race/ethnicity"]])

education_values = ['some high school', 'high school', 'some college',
                    "associate's degree", "bachelor's degree","master's degree" ]
gender_values = ["male", "female"]
lunch_values = x_train["lunch"].unique()
test_values = x_train["test preparation course"].unique()


# "Ordinal" Missing Value
ord_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('encoder', OrdinalEncoder(categories=[education_values, gender_values, lunch_values, test_values]))
])
# result = ord_transformer.fit_transform(x_train[["parental level of education"]])


preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num_transformer, ["reading score", "math score"]),
    ("ord_feature", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nom_feature", nom_transformer, ["race/ethnicity"]),
])

# reg = Pipeline(steps=[
#   ("preprocessor", preprocessor),
#  ("model", LinearRegression())
#])

# reg = Pipeline(steps=[
#     ("preprocessor", preprocessor),
#     ("model",RandomForestRegressor())
# ])

# params = {
#     "preprocessor__num_feature__imputer__strategy": ["median", "mean"],
#     "model__n_estimators": [100, 200, 300],
#     "model__criterion": ["absolute_error", "poisson", "squared_error"],
#     "model__max_depth": [3, 5, 7]
# }
#
# grid = RandomizedSearchCV(estimator=reg, param_distributions=params, n_iter=10, cv=4, scoring="neg_mean_absolute_error", verbose=2, n_jobs=4)
# grid.fit(x_train, y_train)
#
# # result = reg.fit_transform(x_train)
# # reg.fit(x_train, y_train)
# y_predict = grid.predict(x_test)
#
# for i, j in zip(y_predict, y_test):
#     print("Predicted value: {}. Actual value: {}".format(i, j))
#
# # Model evaluation
# print("MAE: {}".format(mean_absolute_error(y_test, y_predict))) # the smaller the better
# print("MSE: {}".format(mean_squared_error(y_test, y_predict))) # the smaller the better
# print("R2 Score: {}".format(r2_score(y_test, y_predict))) # good ~1, bad ~0


reg = LazyRegressor(verbose=0, ignore_warnings=None, custom_metric=None)
models, predictions = reg.fit(x_train, x_test, y_train, y_test)

