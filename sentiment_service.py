"""
Sentiment Analysis Service

This module provides sentiment analysis functionality using a pre-trained SVM model.
The model was trained on IMDb movie review data and can classify reviews as positive or negative.
"""

import joblib
import string
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the trained model and vectorizer
MODEL_PATH = r"D:\Courses\GP-Cinemate\Flask\Machine-Learning\preprocessing\sentiment_model.joblib"
VECTORIZER_PATH = r"D:\Courses\GP-Cinemate\Flask\Machine-Learning\preprocessing\tfidf_vectorizer.joblib"

# Global variables to store loaded model and vectorizer
model = None
vectorizer = None

def load_sentiment_model():
    """Load the sentiment analysis model and vectorizer."""
    global model, vectorizer
    
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        if not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(f"Vectorizer file not found at: {VECTORIZER_PATH}")
            
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("Sentiment model and vectorizer loaded successfully.")
        
    except Exception as e:
        print(f"Error loading sentiment model: {e}")
        raise

def preprocess_text(text):
    """
    Preprocess text for sentiment analysis.
    
    Args:
        text (str): Raw review text
        
    Returns:
        str: Cleaned and preprocessed text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def predict_sentiment(review_text):
    """
    Predict sentiment of a movie review.
    
    Args:
        review_text (str): The movie review text
        
    Returns:
        dict: Dictionary containing sentiment prediction and confidence or error message
    """
    global model, vectorizer
    
    # Input validation
    if not review_text:
        return {"error": "Missing 'review' field in request"}
    
    if not isinstance(review_text, str):
        return {"error": "Review must be a string"}
    
    if len(review_text.strip()) == 0:
        return {"error": "Review text cannot be empty"}
    
    # Load model if not already loaded
    if model is None or vectorizer is None:
        try:
            load_sentiment_model()
        except Exception as e:
            return {"error": f"Failed to load sentiment model: {str(e)}"}
    
    try:
        # Preprocess the review text
        cleaned_review = preprocess_text(review_text)
        
        if not cleaned_review:
            return {
                "error": "Review text is empty or invalid after preprocessing"
            }
        
        # Vectorize the preprocessed text
        review_vector = vectorizer.transform([cleaned_review])
        
        # Make prediction
        prediction = model.predict(review_vector)[0]
        probabilities = model.predict_proba(review_vector)[0]
        
        # Get confidence score for the predicted class
        confidence = probabilities[prediction]
        
        # Convert prediction to sentiment label
        sentiment = 'positive' if prediction == 1 else 'negative'
        
        return {
            "sentiment": sentiment,
            "confidence": float(confidence),
            "probabilities": {
                "negative": float(probabilities[0]),
                "positive": float(probabilities[1])
            }
        }
        
    except Exception as e:
        return {
            "error": f"Error during sentiment prediction: {str(e)}"
        }

# Initialize NLTK data on module import
def initialize_nltk():
    """Download required NLTK data if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt_tab')

def analyze_review_sentiment(review_text):
    """
    Main function to analyze sentiment of a movie review.
    This function serves as the primary interface for the Flask endpoint.
    
    Args:
        review_text (str): The movie review text
        
    Returns:
        dict: Dictionary containing sentiment prediction and confidence or error message
    """
    return predict_sentiment(review_text)

# Initialize NLTK data on module import
initialize_nltk()