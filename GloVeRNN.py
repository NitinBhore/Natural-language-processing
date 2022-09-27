##############    GloVe-Contextualized Vectors uisng RNN    #######################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
plt.style.use('ggplot')

import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.regularizers import l1
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Embedding,RNN, SimpleRNN,Dense,SpatialDropout1D
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
stop=set(stopwords.words('english'))

class GloVe():
  """
  GloVe stands for Global Vectors for word representation. It is an unsupervised 
  learning algorithm. Global Vectors  generate word embeddings by aggregating 
  global word co-occurrence matrices from a given corpus.

  Args:
  embedding_dict : Embedding dictionary
  df : Dataframe name
  column  : Text column name

  Returns: 
    model : GloVe-Contextualized Vectors with SimpleRNN model
  
  """
  def __init__(self, embedding_dict, df, sms):
    """ Inits the Preprocessing """
    self.embedding_dict = embedding_dict
    self.df = df
    self.sms = sms

  # clean text
  def clean_text(self, text):
    """Clean the text"""
    text = re.sub('[^a-zA-Z]', ' ', text)  
    text = text.lower()  
    text = text.split(' ')      
    text = [w for w in text if not w in set(stopwords.words('english'))] 
    text = ' '.join(text)            
    return text

  # create the corpus GloVe 
  def create_corpus(self, df):
      """ create the corpus GloVe """
      corpus=[]
      for tweet in tqdm(df['clean_sms']):
          words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]
          corpus.append(words)
      return corpus
  
  def run_all(self):
    """ Run all the methods as per the requirements """
    df['clean_sms'] = df['sms'].apply(lambda x : self.clean_text(x))

    # padding
    MAX_LEN=10
    tokenizer_obj=Tokenizer()

    corpus=self.create_corpus(df)
    tokenizer_obj.fit_on_texts(corpus)
    sequences=tokenizer_obj.texts_to_sequences(corpus)

    email_pad = pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')

    word_index=tokenizer_obj.word_index

    # Embedding
    num_words=len(word_index)+1
    embedding_matrix=np.zeros((num_words,50))

    for word, i in tqdm(word_index.items()):
        if i > num_words:
            continue
        
        emb_vec=embedding_dict.get(word)
        if emb_vec is not None:
            embedding_matrix[i]=emb_vec

    # Dataset split
    X_train,X_val, y_train, y_val = train_test_split(email_pad,df.spam, test_size=.2, random_state=2)

    # Create Model.
    model=Sequential()

    embedding_layer=Embedding(num_words,50,embeddings_initializer=Constant(embedding_matrix),
                      input_length=MAX_LEN,trainable=False)

    model.add(embedding_layer)
    model.add(SpatialDropout1D(0.2))
    model.add(SimpleRNN(64, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
    model.add(tf.keras.layers.SimpleRNN(32,return_sequences=True))
    model.add(tf.keras.layers.SimpleRNN(16))
    model.add(tf.keras.layers.Dense(16, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    optimzer=Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['acc'])
    # model.summary()

    #Fitting The Model
    history=model.fit(X_train,y_train,batch_size=32,epochs=10,validation_data=(X_val,y_val),verbose=1)

    y_predicted = model.predict(X_val)
    y_predicted = y_predicted.flatten()

    y_predicted = np.where(y_predicted > 0.5, 1, 0)

    print(classification_report(y_val, y_predicted))
    
    return model


embedding_dict={}
with open('glove.6B.50d.txt','r') as f:
    for line in f:
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
f.close()

df = pd.read_csv("SMSCollection.csv")
df['spam']=df['Class'].replace({'ham':0,'spam':1})
df = df.head(100)
df.head()


GloVeExe = GloVe(embedding_dict, df, 'sms')
model = GloVeExe.run_all()