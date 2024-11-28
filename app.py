
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('best_xgb_model.joblib')
scaler = joblib.load('scaler.joblib')

# Load stop words and lemmatizer (only needs to be done once)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # This function takes in the text data from the user and preprocess it similar to 
    # your cleaning function
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(
        r'[^\w\s]|[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002700-\U000027BF]',
        '', text
    )
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

def get_word2vec_embedding(text,model,vector_size):
    # This function takes preprocessed text and generates its embeddings. You need to 
    # modify this section according to your code.
    tokenized_reviews = [text.split()]
    return np.mean([model.wv[token] for token in tokenized_reviews[0] if token in model.wv.key_to_index], axis = 0)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        tweet = request.form['tweet']
        cleaned_tweet = preprocess_text(tweet)
        # Assuming you have a function to generate embeddings (modify this part)
        try:
            embedding = get_word2vec_embedding(cleaned_tweet,model, 100) # Get embedding
            embedding = scaler.transform(embedding.reshape(1, -1)) # Scale the embedding
            prediction = model.predict(embedding)[0]
        except Exception as e:
            prediction = "Error: Could not process tweet. " + str(e)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug = True)