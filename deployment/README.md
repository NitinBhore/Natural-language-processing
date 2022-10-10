# Deployment
This document describes the components like preprocessing, training, evaluation and deploymnet of the classification model of the Natural language processing. Text classification also known as text tagging or text categorization is the process of categorizing text into organized groups. By using Natural Language Processing (NLP), text classifiers can automatically analyze text and then assign a set of pre-defined tags or categories based on its content.

**Inference** repository consists of the python scripts like app.py, prediction.py, requirements.txt and task.py

***

      **task.py** file used to train the BERT classifier model

      Args:
        * bert_preprocess : preprocess URL link
        * bert_encoder : encoder URL link
        * df_balanced : dataframe name
        * text_column : text column name
        * label_column : target column name
        * epochs : number of epochs
        * path : location to save the model

      Returns:
        * model : BERT model

      **prediction.py** file used to get the prediction of the classifier model

      Args:
        * model: Model for prediction

      Returns:
        * prediction_result : prediction whether sms is Ham or Spam

      **requirements.txt** file listing all the dependencies for the deployment of the classification

      **app.py** Flask is a Python-based microframework used for deploying the classifier

      **Docker** The Docker file holds the information of the docker container for the classifier model


***

**Docker** The Docker file holds the information of the docker container for the classifier model
