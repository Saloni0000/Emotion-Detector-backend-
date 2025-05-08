# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import re
import nltk
import pickle
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware



# Download required resources
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and preprocessing tools
lg = pickle.load(open('logistic_regresion.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
lb = pickle.load(open('label_encoder.pkl', 'rb'))

# Define input schema using Pydantic
class TextInput(BaseModel):
    text: str

# Initialize FastAPI app
app = FastAPI(title="Emotion Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 use your actual GitHub Pages URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Text preprocessing function
def clean_text(text: str) -> str:
    stemmer = PorterStemmer()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return " ".join(text)

# Prediction endpoint
@app.post("/predict", response_model=Dict[str, str])
def predict_emotion(input_data: TextInput):
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    cleaned = clean_text(input_data.text)
    vectorized = tfidf_vectorizer.transform([cleaned])

    prediction = lg.predict(vectorized)[0]
    predicted_emotion = lb.inverse_transform([prediction])[0]
    probability = np.max(lg.predict_proba(vectorized))

    return {
        "emotion": predicted_emotion,
        "confidence": f"{probability:.4f}"
    }
