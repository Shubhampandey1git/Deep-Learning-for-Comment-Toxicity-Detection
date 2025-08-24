# 💬 Deep Learning for Comment Toxicity Detection with Streamlit

This project provides an end-to-end solution for detecting and classifying toxic online comments. It leverages both traditional machine learning and deep learning models, wrapped in an interactive web application built with Streamlit for real-time analysis.

---

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [How to Run the Application](#how-to-run-the-application)
- [Model and Data Details](#model-and-data-details)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Introduction

Online toxicity is a significant problem that can hinder constructive conversations. This project aims to address this issue by building a robust system to identify and flag toxic comments. The core of this project is a Streamlit application that allows users to analyze comments in real-time or in bulk by uploading a CSV file. The included Jupyter Notebook (`test.ipynb`) offers a detailed walkthrough of the entire machine learning pipeline, from data preprocessing to model training and evaluation.

---

## Features

*   **Interactive Web Interface:** A user-friendly dashboard for easy interaction with the models.
*   **Real-Time Prediction:** Instantly classify the toxicity of a single comment.
*   **Bulk Prediction:** Upload a CSV file with a `comment_text` column to get predictions for multiple comments.
*   **Model Selection:** Choose between a Convolutional Neural Network (CNN) and a Long Short-Term Memory (LSTM) model for predictions.
*   **Comprehensive Jupyter Notebook:** A step-by-step guide covering the entire model development process.

---

## Project Structure

```
.
├── models/
│   ├── cnn_model.h5
│   ├── lstm_model.h5
│   └── toxicity_model.pkl
├── data/
│   ├── train.csv
│   └── test.csv
├── app.py
├── test.ipynb
└── README.md
```

---

## Technologies Used

*   **Python:** The core programming language.
*   **Streamlit:** For building the interactive web application.
*   **TensorFlow (Keras):** For developing and training the deep learning models (CNN and LSTM).
*   **Scikit-learn:** For the baseline Logistic Regression model and TF-IDF vectorization.
*   **Pandas:** For data manipulation and analysis.
*   **NLTK:** For natural language processing tasks like stop-word removal and lemmatization.
*   **Joblib & Pickle:** For model persistence.

---

## Setup and Installation

Follow these steps to get the project running on your local machine.

**1. Clone the repository:**

```bash
git clone https://github.com/Shubhampandey1git/Deep-Learning-for-Comment-Toxicity-Detection.git
cd Deep-Learning-for-Comment-Toxicity-Detection
```

**2. Create and activate a virtual environment (recommended):**

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

**3. Install the required libraries:**
Create a `requirements.txt` file with the following content and then run `pip install -r requirements.txt`.

```
tensorflow
streamlit
pandas
scikit-learn
nltk
joblib
```

**4. Download NLTK data:**
Run the following commands in a Python interpreter:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## How to Run the Application

Once the setup is complete, you can launch the Streamlit application with this command:

```bash
streamlit run app.py
```

Your default web browser will open a new tab with the application running.

### Using the App

*   **Single Comment Prediction:**
    1.  Type or paste a comment into the text area.
    2.  Select either the "CNN" or "LSTM" model.
    3.  Click "Predict" to view the toxicity probability and classification.

*   **Bulk Prediction from CSV:**
    1.  Use the "Browse files" button to upload a CSV file.
    2.  The CSV file must contain a column named `comment_text`.
    3.  The app will display a preview of the predictions and provide a button to download the results.

---

## Model and Data Details

### Data

The models were trained on the `train.csv` dataset, which contains comments labeled for different types of toxicity. For this project, these labels were consolidated into a single binary target: `1` for any toxic label and `0` for non-toxic.

### Preprocessing

The text data undergoes several preprocessing steps:
1.  **Lowercasing:** All text is converted to lowercase.
2.  **Punctuation and Special Character Removal:** Non-alphabetic characters are removed.
3.  **Stop-word Removal:** Common English stop-words are filtered out.
4.  **Lemmatization:** Words are reduced to their root form.

### Models Trained

The `test.ipynb` notebook details the training of three models:

1.  **Logistic Regression:** A baseline model using TF-IDF features.
2.  **LSTM (Long Short-Term Memory):** An RNN architecture well-suited for sequential data like text.
3.  **CNN (Convolutional Neural Network):** A 1D CNN for extracting local features from word embeddings.

---

## Acknowledgments

*   The developers of the powerful open-source libraries used in this project.
*   The community providing the dataset for this important research area.

---

## 🗂️ Data sets
* This project was created during and Internship.
* If you want to use the data that I have used, you can contact me.

---

## 🙋‍♂️ Author

**Shubham Pandey**
📧 [Email Me](mailto:shubhamppandey1084@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/shubham-pandey-6a65a524a/) • [GitHub](https://github.com/Shubhampandey1git)