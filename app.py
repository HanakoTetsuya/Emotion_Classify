import pandas as pd
import numpy as np
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import streamlit as st

rad=st.sidebar.radio("Navigation Menu",["Gogatubyo"])

#displays all the available disease prediction options in the web app
if rad=="Home":
    st.title("Mental Disorder Predictions App")
    st.text("The Following Disease Predictions Are Available ->")
    st.text("1. Gogatsubyo")

data = pd.read_csv('dataset.csv')
data=data.dropna(how='any')

X = data["text"]
y = data['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42, stratify = y)

tfidf = TfidfVectorizer(max_features= 2500, min_df= 2)
X_train = tfidf.fit_transform(X_train).toarray()
X_test = tfidf.transform(X_test).toarray()
model = RandomForestClassifier(n_estimators= 300)
model.fit(X_train, y_train)

#heading over to the Gogatsubyo section
if rad=="Gogatsubyo":
    st.header("Know If You Are Affected By Gogatsubyo")
    st.write("Fill or copy & paste your feeling in SNS")
    raw_text=st.text_area("Text Here")"
    raw_text=tfidf.transform([raw_text]).toarray()
    
    prediction=model.predict(raw_text)[0]
    
    if st.button("Predict"):
        if prediction==1:
            st.warning("You Might Be Affected By Gogatsubyo")
        elif prediction2==0:
            st.success("You Are Safe")
