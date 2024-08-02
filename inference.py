from flask import Flask, render_template, request, redirect, url_for
import flask
import joblib
from sklearn.ensemble import RandomForestClassifier
import json
from flask import Flask,request,Response, jsonify
import pandas as pd

application = Flask(__name__)
pipeline = joblib.load("./saved_models/GBM.sav")
vectorizer = joblib.load("./saved_models/vectorizeGBM.sav") 

def readRequest(text):
    #print("read request is working")
    dictionary = {}

    ndf = pd.DataFrame()
    ndf["text"] = [text]
    testingdata = vectorizer.transform(ndf["text"])
    result= pipeline.predict(testingdata)
    dictionary.update({"Sentiment of tweet is": result[0]})
    
   
    return dictionary

@application.route('/SentimentAnalysis', methods=['POST', 'GET'])
def SentimentAnalysis():
    param=(request.args.get('input',None))
    #text = list(param)
    text = param
    #print(text)
    rt = readRequest(text)
    return jsonify(rt)
    
## Define HomePage
@application.route('/')
def home():
    return render_template('home.html')
    
#
if __name__ == '__main__':
   application.run(host="0.0.0.0",port=9052,debug=False)

"""
lsof -i:9052
kill -9 <PID>
"""

  
# @application.route('/', methods=['POST'])
# def get_data():
#     text = request.form['search']
#     return redirect(url_for('success', name=text))

  
# @application.route('/success/<name>')
# def success(name):
#     return "<xmp>" + str(readRequest(name)) + " </xmp> "


    
# if __name__ == '__main__':
#   application.run(host="0.0.0.0",port=9052,debug=False)
    
    
    
    




    