"""Import library"""
import numpy as np
import pandas as pd
from keras.models import load_model
import tensorflow_text as text # pylint: disable=unused-import

"""
Blueprint for the prediction of the sms (Ham or Spam)
Args:
  model: Model for prediction
  # text: List of the sentences or Dataframe
Returns:
  prediction_result : prediction whether sms is Ham or Spam
""" # pylint: disable=pointless-string-statement

class Prediction(): # pylint: disable=too-few-public-methods
    """ Prediction is nothing but the Blueprint for the prediction of the sms (Ham or Spam)"""
    def __init__(self, model_path):
        """Inits the Preprocessing"""
        self.model = load_model(model_path)
        
    def get_prediction(self, text):
        """Get the predictions"""
        self.text = text
        if type(self.text) != pd.pandas.core.series.Series:  # pylint: disable=unidiomatic-typecheck
            self.text = list(self.text)

        y_predicted = self.model.predict(self.text)
        # y_predicted = y_predicted.flatten()
        y_predicted = np.where(y_predicted > 0.5, 'Ham', 'Spam')
        return y_predicted
