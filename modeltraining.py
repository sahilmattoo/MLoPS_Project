import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import joblib

def train_model(corpus, target, model, model_name): 
    X_train,X_test,Y_train, Y_test = train_test_split(corpus, target, test_size=0.25, random_state=30)
    vectorizer= TfidfVectorizer()
    tf_x_train = vectorizer.fit_transform(X_train)
    tf_x_test = vectorizer.transform(X_test)
    model.fit(tf_x_train,Y_train)
    filename = "./saved_models/"+model_name+".sav"
    joblib.dump(model, filename)
    y_pred = model.predict(tf_x_test)
    accuracy = accuracy_score(Y_test, y_pred)
    return vectorizer, accuracy

    