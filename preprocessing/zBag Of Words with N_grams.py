#####   Bag Of Words with N_grams   #############

from sklearn.feature_extraction.text import CountVectorizer

def func_baf_of_words(docList, ngram_range=(2,2)):
  vectorizer = CountVectorizer(stop_words='english', ngram_range= ngram_range)
  X = vectorizer.fit_transform(docList)
  df_bow = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names_out())
  return df_bow

docList = ['the man went out for a walk', 'the children sat around the fire', 'Game of Thrones is an amazing tv series!', 'Game of Thrones is the best tv series!', 'Game of Thrones is so great']
result = func_baf_of_words(docList, (3,3))
result