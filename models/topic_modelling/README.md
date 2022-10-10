# Topic Modeling 
This document describes the components like preprocessing, training and evaluation of the topic modeling model of Natural language processing.
topic_modelling - Topic modeling is an unsupervised machine learning technique that's capable of scanning a set of documents, detecting word and phrase patterns within them, and clustering word groups and similar expressions that best characterize a set of documents.

**topic_modelling.py** file consists of the training and evaluation of the topic modeling model.


Args:
  * df : Dataframe name
  * text_column : text column name
  * num_topics  : number f 
  * chunksize=100 : Size pf the chunk (By default is 100)
  * passes=10 : Number of passes (By default is 10)

Returns: 
  * model : topic LDA model
