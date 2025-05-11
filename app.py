from flask import Flask, render_template, request
import pickle
import re
import string
from scipy.sparse import hstack
import numpy as np


app = Flask(__name__)

# Load models
model = pickle.load(open("logistic_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]|https?://\S+|www\.\S+|<.*?>+|[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    tokens = re.findall(r'\b\w+\b', text)
    basic_stopwords = set([
        "the", "and", "is", "in", "it", "of", "to", "a", "with", "for", "on",
        "that", "this", "as", "are", "was", "at", "by", "an", "be", "have", "from"
    ])
    tokens = [word for word in tokens if word not in basic_stopwords and len(word) > 2]
    return ' '.join(tokens)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        input_text = request.form["news"]
        clean_text = preprocess_text(input_text)
        text_length = len(clean_text.split())
        text_vector = vectorizer.transform([clean_text])
        combined_features = hstack([text_vector, [[text_length]]])
        prediction = model.predict(combined_features)[0]
        result = "Real News" if prediction == 1 else "Fake News"
    return render_template("index.html", result=result)

import os

if __name__ == "__main__":
    app.run(debug=True)
