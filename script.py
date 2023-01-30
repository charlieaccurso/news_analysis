import pandas as pd
import numpy as np
from articles import articles
from preprocessing import preprocess_text

# import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

# view article
print(articles[0])

# preprocess articles
processed_articles= [preprocess_text(article) for article in articles]
print(processed_articles[0])

# initialize and fit CountVectorizer
vectorizer= CountVectorizer()
counts= vectorizer.fit_transform(processed_articles)

# convert counts to tf-idf
transformer= TfidfTransformer(norm=None)
tfidf_scores_transformed= transformer.fit_transform(counts)

# initialize and fit TfidfVectorizer
# we want to confirm that the TfidfTransformer gives the same results as directly using the TfidfVectorizer
vectorizer= TfidfVectorizer(norm=None)
tfidf_scores= vectorizer.fit_transform(processed_articles)

# check if tf-idf scores are equal
if np.allclose(tfidf_scores_transformed.todense(), tfidf_scores.todense()):
  print(pd.DataFrame({'Are the tf-idf scores the same?':['YES']}))
else:
  print(pd.DataFrame({'Are the tf-idf scores the same?':['No, something is wrong :(']}))

# get vocabulary of terms
try:
  feature_names = vectorizer.get_feature_names()
except:
  pass

# get article index
try:
  article_index = [f"Article {i+1}" for i in range(len(articles))]
except:
  pass

# create pandas DataFrame with word counts
try:
  df_word_counts = pd.DataFrame(counts.T.todense(), index=feature_names, columns=article_index)
  print(df_word_counts)
except:
  pass

# create pandas DataFrame(s) with tf-idf scores
try:
  df_tf_idf = pd.DataFrame(tfidf_scores_transformed.T.todense(), index=feature_names, columns=article_index)
  print(df_tf_idf)
except:
  pass

try:
  df_tf_idf = pd.DataFrame(tfidf_scores.T.todense(), index=feature_names, columns=article_index)
  print(df_tf_idf)
except:
  pass

# get highest scoring tf-idf term for each article
for i in range(1, 11):
  print(df_tf_idf[[f'Article {i}']].idxmax())

"""
TOPICS:
  Article 1    fare
  dtype: object
  Article 2    hong (hong kong)
  dtype: object
  Article 3    sugar
  dtype: object
  Article 4    petrol
  dtype: object
  Article 5    engine
  dtype: object
  Article 6    australia
  dtype: object
  Article 7    car
  dtype: object
  Article 8    railway
  dtype: object
  Article 9    cabinet
  dtype: object
  Article 10    china
  dtype: object
"""
