
import re
import nltk
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from gensim.models import Word2Vec
from joblib import dump, load

# Download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()



# Define text preprocessing function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove @ mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(
        r'[^\w\s]|[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002700-\U000027BF]',
        '', text)  # Remove emojis and special characters
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)



# Load the dataset:
# Load models and pre-trained objects
WORD2VEC_MODEL_PATH = "word2vec_model.joblib"
XGB_MODEL_PATH = "best_xgb_model.joblib"
SCALER_PATH = "scaler.joblib"

# Load pre-trained models and scaler
try:
    word2vec_model = joblib.load(WORD2VEC_MODEL_PATH)
    model = joblib.load(XGB_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    vector_size = word2vec_model.vector_size
except FileNotFoundError as e:
    st.error(f"Error: {e}")
    st.stop()



# Text preprocessing function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove @ mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(
        r'[^\w\s]|[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002700-\U000027BF]',
        '', text)  # Remove emojis and special characters
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)



def get_avg_word2vec(tokens, model, vector_size):
    valid_tokens = [token for token in tokens if token in model.wv.key_to_index]
    if not valid_tokens:
        return np.zeros(vector_size) 
    return np.mean([model.wv[token] for token in valid_tokens], axis = 0)



# Streamlit app interface
st.title('Hate Speech Detection')
st.markdown('### Classify Text as Hate Speech, Offensive Language, or Neutral.')

# Text input
user_input = st.text_area('Enter text to classify:', '')



# Prediction logic
if st.button('Classify'):
    if user_input:
        # Preprocess the user input
        cleaned_text = clean_text(user_input)
        tokens = cleaned_text.split()

        # Get Word2Vec embeddings for the input text
        embedding = get_avg_word2vec(tokens, word2vec_model, 100)

        # Scale the embeddings for the model input
        scaled_embedding = scaler.transform([embedding])

        # Predict the class using the XGBoost model
        prediction = model.predict(scaled_embedding)
        if prediction == 0:
            st.success('This text is **Hate Speech** (Label: 0).')
        elif prediction == 1:
            st.warning('This text is **Offensive Language** (Label: 1).')
        else:
            st.info('This text is **Neutral** (Label: 2).')
    else:
        st.warning('Please enter a message to classify.')
