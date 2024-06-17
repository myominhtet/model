from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import joblib
import re
from typing import List
from transformers import PreTrainedTokenizerFast
from icu import Transliterator
from myanmartools import ZawgyiDetector

app = FastAPI()
converter = Transliterator.createInstance('Zawgyi-my')
# Load models and tokenizer
model = joblib.load('voting.pkl')
encoder = joblib.load('lable_encode.pkl')

# Define the dummy function to be used in vectorizer
def dummy(text):
    """
    A dummy function to use as tokenizer for TfidfVectorizer.
    It returns the text as it is since we already tokenized it.
    """
    return text

# Load vectorizer with the correct dummy tokenizer function
vectorizer = joblib.load('tfidf_vectorizer.pkl')
vectorizer.set_params(tokenizer=dummy)

tokenizer = PreTrainedTokenizerFast.from_pretrained('tokenize/')
detector = ZawgyiDetector()

# Helper functions
def preprocess_text(text):
    if detector.get_zawgyi_probability(text) > 0.9:
        unicode_text = converter.transliterate(text)
    else:
        unicode_text = text
    unicode_text = re.sub(r'။', '', unicode_text)
    unicode_text = re.sub(r'၊', '', unicode_text)
    unicode_text = unicode_text.replace('See more', '')
    unicode_text = unicode_text.replace('#', '')
    unicode_text = re.sub(r'[?./|@0-9]', '', unicode_text)
    unicode_text = re.sub(r'[\b[၀-၉]+\b','',unicode_text)
    unicode_text=unicode_text.strip()
    return unicode_text

# Store predictions
stored_predictions = {}

class TextBatch(BaseModel):
    texts: List[str]

@app.post("/predict/")
async def predict(text_batch: TextBatch):
    try:
        # Preprocess each text in the batch
        preprocessed_texts = [preprocess_text(text) for text in text_batch.texts]
        token = [tokenizer.tokenize(text) for text in preprocessed_texts]
        # Transform texts using the vectorizer
        text_vectors = vectorizer.transform(token)
        # Make predictions using the model
        predictions = model.predict(text_vectors)
        # Decode predictions
        decoded_predictions = encoder.inverse_transform(predictions)
        # Store predictions
        stored_predictions['predictions'] = decoded_predictions.tolist()
        return {"predictions": decoded_predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_predictions/")
async def get_predictions():
    if 'predictions' in stored_predictions:
        return stored_predictions['predictions']
    else:
        raise HTTPException(status_code=404, detail="Predictions not found")
