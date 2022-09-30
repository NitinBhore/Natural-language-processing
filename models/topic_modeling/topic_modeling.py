"""Topic Modeling"""

# ! pip install pyLDAvis

# import libraries
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string, pprint
import spacy
# gensim for LDA
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
Topic modeling is an unsupervised machine learning technique that's capable of 
scanning a set of documents, detecting word and phrase patterns within them, 
and automatically clustering word groups and similar expressions that best 
characterize a set of documents.

Args:
  df : Dataframe name
  text_column : text column name
  num_topics  : number f 
  chunksize : Size pf the chunk (By default is 100)
  passes=10 : Number of passes (By default is 10)

Returns: 
  model : topic LDA model

"""


class TopicModelling():
    def __init__(self, df, text_column, num_topics, passes=10):
        """Init the Preprocessing"""
        self.df = df
        self.text_column = text_column
        self.num_topics = num_topics
        self.passes = passes

    def sent_to_words(self, sentences, deacc=True):  # deacc=True removes punctuations
        """tokenize using gensim simple_preprocess"""
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence)))

    def remove_stopwords(self, texts, stop_words):
        """remove stopwords"""
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    # perform the lemmatization
    def lemmatization(self, texts, spacy_en_model, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = spacy_en_model(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    def run_all(self):
        """Run all the methods as per the requirements"""
        # convert to list
        data = self.df[self.text_column].values.tolist()
        data_words = list(self.sent_to_words(data))

        # create list of stop words
        # string.punctuation (from the 'string' module) contains a list of punctuations

        stop_words = stopwords.words('english') + list(string.punctuation)

        # remove stop words
        data_words_nostops = self.remove_stopwords(data_words, stop_words)

        # initialize spacy 'en' model, use only tagger since we don't need parsing or NER
        # python3 -m spacy download en
        spacy_en_model = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

        # lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = self.lemmatization(data_words_nostops, spacy_en_model,
                                             allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        # create dictionary and corpus
        # create dictionary
        id2word = corpora.Dictionary(data_lemmatized)

        # Create corpus
        corpus = [id2word.doc2bow(text) for text in data_lemmatized]

        # Build LDA model
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=self.num_topics,
                                                    random_state=100, update_every=1,
                                                    passes=self.passes, alpha='auto', per_word_topics=True)

        return lda_model


df = pd.read_csv("Airbnb_Texas_Rentals.csv")

topicModelling = TopicModelling(df, 'description', 4, 10)
lda_model = topicModelling.run_all()

# print the topics
pprint.pprint(lda_model.print_topics())
