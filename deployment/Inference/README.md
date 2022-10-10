# Inference

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
