## ! pip install tensorflow_text
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
import seaborn as sn


"""
Blueprint of Bidirectional Encoder Representations from Transformers. BERT is a transformer-based machine learning technique for
natural language processing.

Args:
  bert_preprocess : preprocess URL link
  bert_encoder : encoder URL link
  df_balanced : dataframe name
  text_column : text column name
  label_column : target column name
  epochs : number of epochs
  path : location to save the model

Returns: 
  model : BERT model

"""

class BERT():
  def __init__(self, bert_preprocess, bert_encoder, df_balanced, text_column, label_column, epochs, path):
    """Inits the BERT"""
    self.bert_preprocess = bert_preprocess
    self.bert_encoder = bert_encoder
    self.df_balanced = df_balanced
    self.text_column = text_column
    self.label_column = label_column
    self.epochs = epochs
    self.path = path
  
  def func_bert(self, bert_preprocess, bert_encoder, df_balanced, text_column, label_column, epochs, path):
    """Perform the Bidirectional Encoder Representations from Transformers"""
    
    # Split it into training and test data set
    X_train, X_test, y_train, y_test = train_test_split(df_balanced[text_column], df_balanced[label_column], stratify=df_balanced[label_column])

    # Bert layers
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)

    # Neural network layers
    lay = tf.keras.layers.Dense(64, activation='relu',  name="dense1")(outputs['pooled_output'])
    lay = tf.keras.layers.Dropout(0.2, name="dropout1")(lay)
    lay = tf.keras.layers.Dense(32, activation='relu', name="dense2")(lay)
    lay = tf.keras.layers.Dropout(0.2, name="dropout")(lay)
    lay = tf.keras.layers.Dense(1, activation='sigmoid', name="output2")(lay)

    # Use inputs and outputs to construct a final model
    model = tf.keras.Model(inputs=[text_input], outputs = [lay])

    # print sumary
    model.summary()

    model.compile(optimizer='adam',
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    
    # train the model
    model.fit(X_train, y_train, epochs=epochs, )
    
    y_predicted = model.predict(X_test)
    y_predicted = y_predicted.flatten()
    y_predicted = np.where(y_predicted > 0.5, 1, 0)

    print(classification_report(y_test, y_predicted)) 
    cm = confusion_matrix(y_test, y_predicted)
    sn.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')

    # save model
    model.save(path)
    print("\n Model saved on location: ", path)

    return model
 
  def run_all(self):
    """Run all the methods as per the requirements"""
    model = self.func_bert(self.bert_preprocess, self.bert_encoder, self.df_balanced, self.text_column, self.label_column, self.epochs, self.path)
    return model

