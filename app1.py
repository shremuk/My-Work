
"""
Created on Mon Jan  6 16:36:31 2020

@author: ShreyaM
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
os.chdir('C:\Shreya\HR Analytics\Flask')

app = Flask(__name__, template_folder='template')
model = pickle.load(open("LR_v1.pkl", "rb"))

@app.route('/', methods=['GET'])
def home():
    print('home')
    return render_template('front1.html')

@app.route('/predict', methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    #int_features.encode('utf-8')
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template(output)

@app.route('/results', methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    
    if output == 0:
        print('Candidate may join')
    else:
        print('Candidate may drop')
    return jsonify(output)

if __name__ == "__main__":
    app.run(port = 5000, debug=True)
    
    