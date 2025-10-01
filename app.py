import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFDistilBertModel
import re, string

from transformers import AutoTokenizer, TFAutoModel

tokenizer = AutoTokenizer.from_pretrained("tokenizer-20250930T103114Z-1-001")
model = TFAutoModel.from_pretrained("fake_news_model-20250930T103132Z-1-001")
# E:\project\Fake news detection\tokenizer-20250930T103114Z-1-001

# --- Preprocessing functions ---
def lower(text):
    return text.lower()

def clean_text(text):
    text = re.sub(r'\d+', '', text)
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_space(text):
    return re.sub(r'\s+', ' ', text).strip()

def preprocess(text):
    return remove_space(clean_text(lower(text)))

# --- Prediction function ---
def predict_fake_news(text):
    cleaned_text = preprocess(text)
    encoding = tokenizer(
        [cleaned_text],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="tf"
    )

    try:
        prediction = model.predict(
            {"input_ids": encoding["input_ids"], "attention_mask": encoding["attention_mask"]}
        )
    except Exception:
        prediction = model.predict(
            [encoding["input_ids"], encoding["attention_mask"]]
        )

    # Convert logits to probabilities
    probs = tf.nn.softmax(prediction, axis=1).numpy()[0]
    predicted_class = int(tf.argmax(probs).numpy())
    confidence = probs[predicted_class] * 100

    label = "Real News " if predicted_class == 1 else "Fake News "
    return f"{label} (Confidence: {confidence:.2f}%)"

# --- Streamlit UI ---
st.title(" Fake News Detection App")
st.write("Enter a news article below to check if it's real or fake:")

user_input = st.text_area("News Article")

if st.button("Check"):
    if user_input.strip():
        result = predict_fake_news(user_input)
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter some text.") 