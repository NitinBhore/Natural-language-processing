# -*- coding: utf-8 -*-
"""text_similarity.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fioIvCwtum2k2roB9X7ysg1Z2yzBz1sl
"""

# !pip install bert-embedding
# !pip install pandas

import os, re, io
import numpy as np
import requests
import pandas as pd
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import spacy
nlp = spacy.load('en_core_web_sm')
from bert_embedding import BertEmbedding
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from sklearn.metrics.pairwise import cosine_similarity

"""Blueprint of Questions similarity using the Bidirectional Encoder 
Representations from Transformers. BERT is a transformer-based machine 
learning technique for natural language processing."""

class QuestionAnswer():
    def __init__(self, data_df, column_name=None):
        """Inits the Preprocessing"""
        self.data_df = data_df  
        self.column = column_name
        self.clean_column_name = f"clean_{self.column}"

    def cleanData(self, data_df, column_name):
      """Clean the data by removing stop words punctuvation"""
      self.data_df.fillna('',inplace=True)
      self.data_df = self.data_df.apply(lambda column: column.astype(str).str.lower(), axis=0)
      self.data_df[self.column] = self.data_df[self.column].apply(lambda row: re.sub(r'^\d+[.]',' ', row))    
      self.data_df[self.column] = self.data_df[self.column].apply(lambda row: re.sub(r'[^A-Za-z0-9\s]', ' ', row)) 
      for idx, question in enumerate(self.data_df[self.column]):
        self.data_df.loc[idx, self.clean_column_name] = remove_stopwords(question)        
      return self.data_df, self.column

    def apply_lemmatization(self, data_df, column_name ):
        """get the root words to reduce inflection of words"""
        lemmatizer = WordNetLemmatizer()    
        self.clean_column_name
        
        for idx, question in enumerate(data_df[column_name]):

            lemmatized_sentence = []
            doc = nlp(question.strip())
            for word in doc:       
                lemmatized_sentence.append(word.lemma_)      
                ## update to the same column
                self.data_df.loc[idx, self.clean_column_name] = " ".join(lemmatized_sentence)

    def run_all(self):
        data_df, column_name = self.cleanData(self.data_df, self.column)
        self.apply_lemmatization(data_df, column_name)    
        return self.data_df

df = pd.read_csv("COVID19_FAQ.csv")
df.head(10)

## pre-process training question data
text_preprocessor = QuestionAnswer(df.copy(), column_name="questions")
clean_df = text_preprocessor.run_all()
clean_df.head(10)

test_query_questions = ["Am I considered a close contact if I was wearing a mask?",
"Is the virus that causes COVID-19 found in feces (stool)?",
"Can the COVID-19 virus spread through sewerage systems?",
"Should I be tested for a current infection?"]

test_df = pd.DataFrame(test_query_questions, columns=["test_questions"])  

## pre-process testing QA data
text_preprocessor = QuestionAnswer(test_df, column_name="test_questions")
query_df = text_preprocessor.run_all()

## get bert embeddings
def func_get_bert_embeddings(sentences):
    bert_embedding = BertEmbedding()
    return bert_embedding(sentences)

question_QA_bert_embeddings_list = func_get_bert_embeddings(clean_df["questions"].to_list())
query_QA_bert_embeddings_list = func_get_bert_embeddings(test_df["test_questions"].to_list())

## store QA bert embeddings in list
question_QA_bert_embeddings = []
for embeddings in question_QA_bert_embeddings_list:
    question_QA_bert_embeddings.append(embeddings[1])

## store query string bert embeddings in list
query_QA_bert_embeddings = []
for embeddings in query_QA_bert_embeddings_list:
    query_QA_bert_embeddings.append(embeddings[1])

## helps to retrieve similar question based of input vectors/embeddings for test query
def func_get_SimilarFAQ(train_question_vectors, test_question_vectors, train_df, train_column_name, test_df, test_column_name):
    similar_question_index = []
    for test_index, test_vector in enumerate(test_question_vectors):
        sim, sim_Q_index = -1, -1
        for train_index, train_vector in enumerate(train_question_vectors):
            sim_score = cosine_similarity(train_vector, test_vector)[0][0]
            
            if sim < sim_score:
                sim = sim_score
                sim_Q_index = train_index

        print(f"Query Question: {test_df[test_column_name].iloc[test_index]}")    
        print(f"Get Question: {train_df[train_column_name].iloc[sim_Q_index]}")
        print("\n")
        
func_get_SimilarFAQ(question_QA_bert_embeddings, query_QA_bert_embeddings, clean_df, "questions", query_df, "test_questions")


