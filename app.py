import numpy as np
from flask import Flask,render_template
import joblib
from flask import request



app = Flask(__name__)
model = joblib.load("argriculture_production.pkl")


@app.route('/')
def prem():
    return render_template('index.html')

@app.route('/',methods=["POST"])
def home1():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    arr = np.array([[data1,data2,data3,data4,data5,data6,data7]],dtype= float)
    pred = model.predict(arr)
    listToStr = ' '.join(map(str, pred))
    print(listToStr)
    return render_template('index.html',prediction_text = f" {listToStr}")

    # return render_template('index.html',prediction_text = f"The Suggested Crop for Given Climatic Codition is :{pred}")

if __name__ == "__main__":
    app.run(debug=True)