import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def format_title(title):
    return title.replace("|", " ").replace("-", "")

df = pd.read_csv('movies.csv', encoding='latin-1', sep='\t', usecols=["title", "genres"])
df["genres"] = df["genres"].apply(format_title)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["genres"])
tfidf_matrix_dense = pd.DataFrame(tfidf_matrix.todense(), index=df["title"],
                            columns=vectorizer.get_feature_names_out())

# Tính cosine similarity - độ tương đồng về thể loại giữa các phim
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim_dense = pd.DataFrame(cosine_similarity(cosine_sim), index=df["title"], columns=df["title"])

input_movie = "Jumanji (1995)"
top_k = 20
result = cosine_sim_dense.loc[input_movie, :]
result = result.sort_values(ascending=False)[:top_k].to_frame("score")