import pickle
from flask import Flask, request, app, jsonify, url_for, render_template

import numpy as np
import pandas as pd

app = Flask(__name__)    # create app, its a starting point

# load model pickel file
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def homepage():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']     #wherenever we hit predict_api, the input will be in json format
    print(data)

    # we need to reshape the datapoints
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))

    # apply model for prediction
    output = regmodel.predict(new_data)   # output will be 2-D array

    print(output[0])
    return jsonify(output[0])


# @app.route('/predict_api', methods=['POST'])
# def predict_api():
#     data = request.json['data']     #wherenever we hit predict_api, the input will be in json format
#     print(data)

#     # we need to reshape the datapoints
#     print(np.array(data)[:, :-1])
#     new_data = scalar.transform(np.array(data)[:, :-1])

#     # apply model for prediction
#     output = regmodel.predict(new_data)   # output will be 2-D array

#     print(output[0])
#     return jsonify(output.tolist())

if __name__ == "__main__":
    app.run(debug=True)
