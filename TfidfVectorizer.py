####  TfidfVectorizer   ############
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def func_tfidf(docList):
  vectorizer = TfidfVectorizer()
  vectors = vectorizer.fit_transform(docList)
  feature_names = vectorizer.get_feature_names()
  dense = vectors.todense()
  denselist = dense.tolist()
  df = pd.DataFrame(denselist, columns=feature_names)
  return df

docList = ['the man went out for a walk', 'the children sat around the fire', 'Game of Thrones is an amazing tv series!', 'Game of Thrones is the best tv series!', 'Game of Thrones is so great']
func_tfidf(docList)