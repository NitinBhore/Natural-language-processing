
import numpy as np
import pandas as pd

"""
Blueprint for the prediction of the sms (Ham or Spam)

Args:
  model: Model for prediction
  text: List of the sentences or Dataframe

Returns: 
  prediction_result : prediction whether sms is Ham or Spam

"""

class Prediction():
    def __init__(self, model, text):
        """Inits the Preprocessing"""
        self.model = model
        self.text = text

    def get_prediction(self):
        """Get the predictions"""
        if type(self.text) != pd.pandas.core.series.Series:
            self.text = list(self.text)

        y_predicted = self.model.predict(self.text)
        # y_predicted = y_predicted.flatten()
        y_predicted = np.where(y_predicted > 0.5, 'Ham', 'Spam')
        return y_predicted
