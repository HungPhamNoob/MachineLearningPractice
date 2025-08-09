import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile


def filter_loc(location):
    result = re.findall("\,\s[A-Z]{2}$", location)
    if result != 0:
        return result[0][2:]
    return location

df = pd.read_excel("final_project.ods", engine="odf", dtype=str)
#print(df.isna().sum()) # check xem có ô rỗng không?
df.dropna(axis=0, inplace=True)
df["location"] = df["location"].apply(filter_loc)


target = "career_level"
x = df.drop(target, axis=1)
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100, stratify=y)

# vectorizer = TfidfVectorizer(stop_words='english')
#
# result = vectorizer.fit_transform(x_train["title"])
#
# print(vectorizer.vocabulary_)
# print(len(vectorizer.vocabulary_))
# print(result)

# Cai thien performence
preprocessor = ColumnTransformer(transformers=[
    ("title_trans", TfidfVectorizer(stop_words='english', ngram_range=(1, 1)), "title"),
    ("location_trans", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("description_trans", TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=0.01, max_df=0.95), "description"),
    ("function_trans", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry_trans", TfidfVectorizer(stop_words='english', ngram_range=(1, 1)), "industry"),
])

# handle_unknown="ignore" để tránh việc báo lỗi bộ train location không chứa thành phố mà bộ test có
# min_df, max_df là threshold để loại bỏ các token thừa thãi - quá hiếm hoặc quá phổ thông

# Cai thien performence
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("feature_selector", SelectKBest(chi2, k=300)), # phai biet tong so feature (words)
    # ("feature_selector", SelectPercentile(chi2, percentile=5))  # ko phai biet tong so feature (words), percentile la % cua tong so features
    ("classifier", RandomForestClassifier())
])


clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print(classification_report(y_test, y_pred))
