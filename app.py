
import os
import re
import joblib
import pandas as pd
import streamlit as st

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="YouTube Comment Sentiment Classifier",
    page_icon="💬",
    layout="wide"
)

# Define local paths relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_logistic_regression_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

# Cache resources so models load only once
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        st.error(f"Artifacts missing! Please check paths:\n- {MODEL_PATH}\n- {VECTORIZER_PATH}")
        return None, None

    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model

# Preprocessing Function (Must match 02_Preprocessing.ipynb logic)
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()                                           # Lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'@\w+|\#', '', text)                          # Remove mentions & hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)                      # Remove punctuation & special chars
    text = re.sub(r'\s+', ' ', text).strip()                     # Clean whitespace
    return text

# Main App Layout
def main():
    st.title("💬 YouTube Comment Classifier")
    st.markdown("Predict sentiment / classification of new user comments using your trained NLP pipeline.")

    # Sidebar
    st.sidebar.header("📌 Project Details")
    st.sidebar.text("Student ID: 22CDS0446")
    st.sidebar.text("Dataset: YouTube Comments")

    vectorizer, model = load_artifacts()

    if vectorizer is None or model is None:
        st.stop()

    # Tabs for Single Prediction & Batch Upload
    tab1, tab2 = st.tabs(["✍️ Single Comment Prediction", "📁 Batch Prediction (CSV)"])

    # ---------------- TAB 1: Single Prediction ----------------
    with tab1:
        st.subheader("Predict Single Text")
        user_input = st.text_area(
            "Enter a comment to analyze:",
            placeholder="Example: This video was super helpful and well explained!",
            height=120
        )

        if st.button("Predict Sentiment", type="primary"):
            if user_input.strip() == "":
                st.warning("Please enter text to classify.")
            else:
                # 1. Clean
                cleaned_input = preprocess_text(user_input)

                # 2. Vectorize
                input_vector = vectorizer.transform([cleaned_input])

                # 3. Predict
                prediction = model.predict(input_vector)[0]

                # Display Result
                st.write("---")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(label="Predicted Output", value=str(prediction))

                with col2:
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(input_vector)[0]
                        classes = model.classes_
                        prob_df = pd.DataFrame({"Class": classes, "Probability": probs})
                        st.dataframe(prob_df, hide_index=True)

    # ---------------- TAB 2: Batch Prediction ----------------
    with tab2:
        st.subheader("Upload CSV File for Batch Predictions")
        uploaded_file = st.file_uploader("Choose a CSV file containing comments", type=["csv"])

        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data:", batch_df.head(3))

            # Select column containing comment text
            text_column = st.selectbox("Select the comment text column:", batch_df.columns)

            if st.button("Run Batch Predictions"):
                with st.spinner("Processing comments..."):
                    # Preprocess and transform
                    clean_texts = batch_df[text_column].astype(str).apply(preprocess_text)
                    batch_vectors = vectorizer.transform(clean_texts)

                    # Predict
                    predictions = model.predict(batch_vectors)
                    batch_df['Predicted_Label'] = predictions

                    st.success("Predictions Complete!")
                    st.dataframe(batch_df.head(10))

                    # Download link
                    csv_data = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Result CSV",
                        data=csv_data,
                        file_name="predicted_comments.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()
