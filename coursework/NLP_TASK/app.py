import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import contractions

# Page setup
st.set_page_config(page_title="Sarcasm Predictor", page_icon="🤖")

st.title("🤖 Real-Time Sarcasm Detection")
st.write("This tool uses a Deep Learning LSTM model to predict if a headline is sarcastic or serious.")

# Load assets
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('./sarcasm_predictor.h5')
    with open('./tokenizer_v2.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_assets()

def clean_input(text):
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Prediction Interface
user_text = st.text_input("Enter a headline:", "")

if st.button("Predict"):
    if user_text:
        processed = clean_input(user_text)
        seq = tokenizer.texts_to_sequences([processed])
        # Use same padding length as model (calculated from 95th percentile in notebook)
        padded = pad_sequences(seq, maxlen=20, padding='post')

        prediction = model.predict(padded)[0][0]

        if prediction > 0.5:
            st.error(f"Prediction: Sarcastic ({prediction:.2%})")
        else:
            st.success(f"Prediction: Not Sarcastic ({1-prediction:.2%})")
    else:
        st.warning("Please enter some text.")

st.sidebar.markdown("### Model Details")
st.sidebar.write("Architecture: LSTM")
st.sidebar.write("Embedding: Trainable Keras Embedding")
st.sidebar.write("Framework: Keras/TensorFlow")
