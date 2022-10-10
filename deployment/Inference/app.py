
"""Import library"""

from flask import Flask, jsonify, request
from prediction import Prediction

# load model
prediction = Prediction(model_path = "model-output")

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])
def predict():
    """Function is used for prediction"""
    # get data ??????????
    data = request.get_json()['text'] #request.list(force=True)  # get_json(force=True) #request.get_json()['text'] # pylint: disable=line-too-long

    # predictions
    result = prediction.get_prediction(data)
    # send back to browser
    output = {'results': int(result[0])}
    return jsonify(results=output)


if __name__ == '__main__':
    app.run(port=5000, debug=True)


# !python app.py
# !curl http://127.0.0.1:5000/ -d "{\"text\": \" i have not go fjisd dhsl \"} " -H 'Content-Type: application/json'
