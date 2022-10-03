
import pandas as pd
from keras.models import load_model
import tensorflow_text as text
from flask import Flask, jsonify, request
import pickle
from prediction import Prediction

# load model
model = load_model("model-output/")

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])
def predict():
    # get data ??????????
    data = request.get_json()['text'] #request.list(force=True)  # get_json(force=True) #request.get_json()['text']

    # # convert data into dataframe
    # data.update((x, [y]) for x, y in data.items())
    # data_df = pd.DataFrame.from_dict(data)

    prediction = Prediction(model, data)

    # predictions
    result = prediction.get_prediction()
    # send back to browser
    output = {'results': int(result[0])}
    return jsonify(results=output)


if __name__ == '__main__':
    app.run(port=5000, debug=True)


# !python app.py
# # !curl http://127.0.0.1:5000/ -d "{\"text\": \" i have not go fjisd dhsl \"} " -H 'Content-Type: application/json'
