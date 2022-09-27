###########   Encoder-Decoder   ##############

import pandas as pd 
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense,Flatten,Conv2D,Conv1D,GlobalMaxPooling1D
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from keras.preprocessing.text import Tokenizer

# Load the input features

def load_train(file_name):
    pd.set_option('display.max_colwidth',None)
    train =pd.read_csv(file_name) 
    train=train.dropna()
    return train
    
train_df = load_train("SMSCollection.csv")
# train_df = train_df.head(100)
train_df['spam']=train_df['Class'].apply(lambda x: 1 if x=='spam' else 0)
X = train_df['sms'] # input
y = train_df[['spam']].values # target /label

sentences_train,sentences_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=11)

tokenizer = Tokenizer(num_words=30000)
tokenizer.fit_on_texts(sentences_train)
X_train = tokenizer.texts_to_sequences(sentences_train)
X_val = tokenizer.texts_to_sequences(sentences_val)

# Adding 1 because of  reserved 0 index
vocab_size = len(tokenizer.word_index) + 1 # (in case of pre-trained embeddings it's +2)                         
maxlen = 131 # sentence length

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)

maxlen = 131
max_features = 50000
embed_size = 131

encoder_inp   = Input(shape=(maxlen,))
encoder_embed = Embedding(max_features,embed_size,input_length=maxlen,trainable=True)(encoder_inp)
encoder_lstm_cell = LSTM(60,return_state='True')
encoder_output,encoder_state_h,encoder_state_c = encoder_lstm_cell(encoder_embed)
#Creating LSTM decoder model and feeding the output states (h,c) of lstm of encoders
decoder_inp   = Input(shape=(maxlen,))
decoder_embed = Embedding(max_features,embed_size,input_length=maxlen,trainable=True)(decoder_inp)
decoder_lstm_cell = LSTM(60,return_sequences='True',return_state=True)
decoder_output,decoder_state_h,decoder_state_c = decoder_lstm_cell(decoder_embed,initial_state=[encoder_state_h,encoder_state_c])
decoder_dense_cell1 = Dense(16,activation='relu')
decoder_d_output    = decoder_dense_cell1(decoder_output)
decoder_dense_cell2 = Dense(1,activation='sigmoid')
decoder_output = decoder_dense_cell2(decoder_d_output)
model = Model([encoder_inp,decoder_inp],decoder_output) 
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#model.summary()
history = model.fit([X_train,X_train],y_train,batch_size=1024,epochs=2)

