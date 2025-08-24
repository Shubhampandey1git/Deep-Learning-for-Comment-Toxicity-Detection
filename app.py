import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import InputLayer, Embedding, Input, Dense
from tensorflow.keras.utils import custom_object_scope
import pickle  # only needed if you later save models
import pandas as pd

def load_resources(train_texts):
    """
    Loads CNN, LSTM models, third model (pkl), and recreates tokenizer.
    train_texts: list or pd.Series of training comments to fit tokenizer
    """
    # ------------------------
    # Recreate tokenizer
    # ------------------------
    tokenizer = Tokenizer(num_words=5000)  # adjust num_words if needed
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
        with open("models/toxicity_model.pkl", "rb") as f:
            third_model = pickle.load(f)
    except FileNotFoundError:
        third_model = None

    return tokenizer, cnn_model, lstm_model, third_model



tokenizer, cnn_model, lstm_model, third_model = load_resources(pd.read_csv("data/train.csv")['comment_text'])

max_len = 100  # same as used in training

# -------------------------
# Prediction helper
# -------------------------
def predict_comment(text, model):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len, padding='post')
    prob = model.predict(pad)[0][0]
    label = "Toxic" if prob >= 0.5 else "Non-toxic"
    return prob, label

# -------------------------
# Streamlit UI
# -------------------------
st.title("💬 Comment Toxicity Detection Dashboard")

# Real-time prediction
st.subheader("Single Comment Prediction")
user_input = st.text_area("Enter a comment:")
model_choice = st.selectbox("Select model", ["CNN", "LSTM"])

if st.button("Predict"):
    if user_input.strip():
        model = cnn_model if model_choice=="CNN" else lstm_model
        prob, label = predict_comment(user_input, model)
        st.write(f"**Toxicity Probability:** {prob:.2f}")
        st.write(f"**Predicted Label:** {label}")

# Bulk predictions
st.subheader("Bulk Prediction from CSV")
uploaded_file = st.file_uploader("Upload a CSV with 'comment_text'", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'comment_text' not in df.columns:
        st.error("CSV must contain a column named 'comment_text'")
    else:
        model = cnn_model if model_choice=="CNN" else lstm_model
        df['toxicity_prob'] = df['comment_text'].apply(lambda x: predict_comment(x, model)[0])
        df['label'] = df['toxicity_prob'].apply(lambda x: 'Toxic' if x>=0.5 else 'Non-toxic')
        st.write(df.head(10))
        st.download_button(
            "Download Predictions",
            data=df.to_csv(index=False),
            file_name="toxicity_predictions.csv",
            mime="text/csv"
        )

# Dataset insights (optional)
st.subheader("Sample Insights")
st.write(f"Uploaded file total comments: {len(df) if uploaded_file else 'N/A'}")
