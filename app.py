import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import InputLayer, Embedding, Input, Dense
from tensorflow.keras.utils import custom_object_scope
import joblib

def load_resources(train_texts):
    """
    Loads CNN, LSTM models, third model (pkl), and recreates tokenizer.
    train_texts: list or pd.Series of training comments to fit tokenizer
    """
    # ------------------------
    # Recreate tokenizer
    # ------------------------
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(train_texts)

    # ------------------------
    # Custom wrappers for old models
    # ------------------------
    def input_layer_wrapper(**kwargs):
        kwargs.pop("batch_shape", None)
        return InputLayer(**kwargs)

    def embedding_wrapper(**kwargs):
        if 'dtype' in kwargs and isinstance(kwargs['dtype'], dict):
            kwargs['dtype'] = 'float32'
        return Embedding(**kwargs)

    # ------------------------
    # Load CNN and LSTM models
    # ------------------------
    with custom_object_scope({"InputLayer": input_layer_wrapper, "Embedding": embedding_wrapper, "Input": Input, "Dense": Dense}):
        cnn_model = load_model("models/cnn_model.h5", compile=False)
        lstm_model = load_model("models/lstm_model.h5", compile=False)

    # ------------------------
    # Load third pkl model if exists
    # ------------------------
    try:
        lr_model = joblib.load("models/toxicity_model.pkl")
        tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    except FileNotFoundError:
        st.error("Logistic Regression model or TF-IDF vectorizer not found. Please ensure they are in the 'models' directory.")
        lr_model, tfidf_vectorizer = None, None

    return tokenizer, cnn_model, lstm_model, lr_model, tfidf_vectorizer

# Load resources



train_df = pd.read_csv("data/train.csv")
tokenizer, cnn_model, lstm_model, lr_model, tfidf_vectorizer = load_resources(train_df['comment_text'].astype(str))

max_len = 100  # same as used in training

# -------------------------
# Prediction helper
# -------------------------
def predict_deep_learning(text, model):
    """Prediction function for CNN and LSTM models."""
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len)
    prob = model.predict(pad)[0][0]
    label = "Toxic" if prob >= 0.5 else "Non-toxic"
    return prob, label

def predict_logistic_regression(text, model, vectorizer):
    """Prediction function for the Logistic Regression model."""
    if not model or not vectorizer:
        return 0.0, "Error"
    # The input to the vectorizer's transform method must be an iterable (like a list)
    text_vectorized = vectorizer.transform([text])
    prob = model.predict_proba(text_vectorized)[0, 1] # Probability of the 'toxic' class
    label = "Toxic" if prob >= 0.5 else "Non-toxic"
    return prob, label


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Comment Toxicity Detection", layout="wide")
st.title("💬 Comment Toxicity Detection Dashboard")
st.markdown("Analyze text for toxicity using different machine learning models.")

# Real-time prediction
st.subheader("Single Comment Prediction")
user_input = st.text_area("Enter a comment:", "this is a sample comment")
model_choice = st.selectbox("Select model", ["Custom Logistic Regression","CNN", "LSTM"])

if st.button("Predict"):
    # if user_input.strip():
    #     st.warning("Please enter a comment.")
    # else:
        prob, label = 0.0, ""
        if model_choice == "Custom Logistic Regression":
            prob, label = predict_logistic_regression(user_input, lr_model, tfidf_vectorizer)
        elif model_choice == "CNN":
            prob, label = predict_deep_learning(user_input, cnn_model)
        elif model_choice == "LSTM":
            prob, label = predict_deep_learning(user_input, lstm_model)

        st.write(f"**Model Used:** `{model_choice}`")
        st.write(f"**Toxicity Probability:** `{prob:.4f}`")
        
        if label == "Toxic":
            st.error(f"**Predicted Label: {label}**")
        else:
            st.success(f"**Predicted Label: {label}**")
# Bulk predictions
st.subheader("Bulk Prediction from CSV")
uploaded_file = st.file_uploader(
    "Upload a CSV with a 'comment_text' column",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'comment_text' not in df.columns:
        st.error("Error: The uploaded CSV must contain a column named 'comment_text'.")
    else:
        st.info(f"Using the **{model_choice}** model for bulk prediction. You can change it in the dropdown above.")
        
        # Ensure comments are strings and handle potential empty comments
        df['comment_text'] = df['comment_text'].astype(str).fillna('')
        
        # Apply the chosen prediction function
        if model_choice == "Logistic Regression":
            predictions = [predict_logistic_regression(text, lr_model, tfidf_vectorizer) for text in df['comment_text']]
        else:
            dl_model = cnn_model if model_choice == "CNN" else lstm_model
            # Note: Predicting one by one is slow for DL models. Batch prediction would be faster.
            predictions = [predict_deep_learning(text, dl_model) for text in df['comment_text']]
            
        df['toxicity_prob'] = [p[0] for p in predictions]
        df['label'] = [p[1] for p in predictions]

        st.write("Prediction Results Preview:")
        st.write(df.head(10))

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name=f"toxicity_predictions_{model_choice.lower()}.csv",
            mime="text/csv",
        )