import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

dataset = pd.read_csv("../../../data/preprocessed/dataset.csv")

bow_vectorizer = CountVectorizer(max_features=1000)  # limitamos vocabulario
X_bow = bow_vectorizer.fit_transform(dataset["review_clean"])

matriz = pd.DataFrame(X_bow.toarray(), columns=bow_vectorizer.get_feature_names_out())
matriz.to_csv("matriz_bow.csv", index=False)

print("Shape BoW:", X_bow.shape)
print(pd.DataFrame(X_bow.toarray(), columns=bow_vectorizer.get_feature_names_out()))
print("Vocabulario ejemplo:", bow_vectorizer.get_feature_names_out())

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(dataset["review_clean"])

print("Shape TF-IDF:", X_tfidf.shape)
print(pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out()))
print("Vocabulario ejemplo:", tfidf_vectorizer.get_feature_names_out())

