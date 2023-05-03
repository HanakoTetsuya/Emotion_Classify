import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import re
import string

import warnings
warnings.filterwarnings('ignore')

from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from scikitplot.metrics import plot_confusion_matrix, plot_roc
import streamlit as st

data = pd.read_csv('dataset.csv')
data=data.dropna(how='any')

X = data["text"]
y = data['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42, stratify = y)

tfidf = TfidfVectorizer(max_features= 2500, min_df= 2)
X_train = tfidf.fit_transform(X_train).toarray()
X_test = tfidf.transform(X_test).toarray()

def train_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

rf = RandomForestClassifier(n_estimators= 300)

emotions_emoji_dict = {"neutral" : "😐", "depressed" : "😔"}

# Main Application
def main():
	st.title("Gogatsubyo Classifier App")
	menu = ["Home"]
	choice = st.sidebar.selectbox("Menu", menu)
	create_page_visited_table()
	create_emotionclf_table()
	if choice == "Home":
		add_page_visited_details("Home", datetime.now())
		st.subheader("Gogatsubyo Detection in Text")

		with st.form(key='emotion_clf_form'):
			raw_text = st.text_area("Type or Copy & Paste Here")
                        raw_text=tfidf.transform([raw_text]).toarray()
			submit_text = st.form_submit_button(label = 'Submit')

		if submit_text:
			col1, col2  = st.columns(2)

			# Apply Function Here
			prediction = rf.predict(raw_text)
			probability = rf.predict_proba(raw_text)
			
			add_prediction_details(raw_text, prediction, np.max(probability), datetime.now())

			with col1:
				st.success("Original Text")
				st.write(raw_text)

				st.success("Prediction")
				emoji_icon = emotions_emoji_dict[prediction]
				st.write("{}:{}".format(prediction, emoji_icon))
				st.write("Confidence:{}".format(np.max(probability)))

			with col2:
				st.success("Prediction Probability")
				# st.write(probability)
				proba_df = pd.DataFrame(probability, columns = rf.classes_)
				# st.write(proba_df.T)
				proba_df_clean = proba_df.T.reset_index()
				proba_df_clean.columns = ["emotions", "probability"]

				fig = alt.Chart(proba_df_clean).mark_bar().encode(x = 'emotions', y = 'probability', color = 'emotions')
				st.altair_chart(fig,use_container_width = True)

	else:
		st.subheader("About")
		add_page_visited_details("About", datetime.now())

if __name__ == '__main__':
	main()
