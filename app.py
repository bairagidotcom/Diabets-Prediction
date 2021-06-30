from flask import Flask , render_template, request
import pickle
import numpy as np

app = Flask(__name__)

diabetes_predict = pickle.load(open('diabetes_predict.pkl', 'rb'))

@app.route('/' , methods=[ 'GET', 'POST'])
def hello_world():
    
    return render_template('index.html')

@app.route('/predict' , methods = ['GET' ,'POST'])
def predict():
    input_val = [float(x) for x in request.form.values()]
    final_val = [np.array(input_val)]
    prediction = diabetes_predict.predict(final_val)
    if prediction[0]==1:
        return render_template('index.html' , pred="Diabetes found !")
    else:
        return render_template('index.html' , pred="Diabetes not found.")


if __name__ == "main":
    app.run(debug=True)