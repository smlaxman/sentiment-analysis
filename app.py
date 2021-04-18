# import Flask class from the flask module
import string

from flask import Flask, request,render_template,jsonify
import joblib
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import json


# Create Flask object to run
app = Flask(__name__)

# Load the model from the file
sentiment_classification_model = joblib.load('model/sentiment_classification_model.pkl')

def validateSEntiments(listOfProductIds):
    sentimentDF= pd.read_pickle("model/Sentimantlookupfile.pkl")
    readyToRecomend={}
    count = 0
    for i in listOfProductIds:
        count += 1
        sentimentDF_lables = sentimentDF.loc[(sentimentDF.id == i) ]
        sentimentDF_lablesPositive=sentimentDF_lables[sentimentDF_lables['class_predicted'] == 1]
        sentimentDF_lablesNegative=sentimentDF_lables[sentimentDF_lables['class_predicted'] == 0]
        PositiveCount=sentimentDF_lablesPositive['class_predicted'].count()
        NegativeCount=sentimentDF_lablesNegative['class_predicted'].count()
        print("======================")
        print(i, PositiveCount, NegativeCount)
        if(PositiveCount > NegativeCount ):
            name=sentimentDF_lables['name'].iloc[0]
            readyToRecomend[i]=name
        if count == 5:
            break
    return  readyToRecomend




@app.route('/', methods=['GET','POST'])
def my_form_post():
    if request.method == "POST":
        userId = request.form.get("UserId")
        print("*********")
        print(userId)
        recomendationDF=pd.read_pickle("model/recomendationlookupfile.pkl")
        d = recomendationDF.loc[userId].sort_values(ascending=False)[0:20]
        print("the product list is ")
        print(d)
        listOfProductIds= (d.index.values.tolist())
        recommendToUser=validateSEntiments(listOfProductIds)
        print(recommendToUser)
        result =" "
        for i in recommendToUser:
            result = result  + i + "->" + recommendToUser[i] + "\n"
        # return("Recommended Products are : \n "+ result)
        return render_template("recommend.html", products=recommendToUser)
    return render_template("home.html")


if __name__ == "__main__":
    # Start Application
    app.run()
