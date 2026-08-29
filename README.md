# YouTube Comment Video Classifier — NLP Project

**Student ID:** 22CDS0446  
**Course:** Natural Language Processing (NLP)  
**Dataset:** YouTube Comments (scraped from 11 videos across multiple channels)

---

## Project Overview

This project builds an **end-to-end NLP pipeline** that classifies YouTube comments into their corresponding **video categories** using supervised machine learning. Given a raw comment, the system predicts which of 11 YouTube videos it belongs to, using a **TF-IDF + Logistic Regression** model served through a **Streamlit web application**.

---

## Project Structure

```
22CDS0446_NLP/
│
├── Dataset/                          # Raw scraped YouTube comment CSVs (12 files)
│
├── models/
│   ├── sentiment_logistic_regression_model.pkl   # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl                      # Fitted TF-IDF vectorizer (5000 features)
│
├── features/                         # Saved feature matrices and labels
│   ├── vectors_x_train_tfidf.pkl     # TF-IDF train vectors
│   ├── vectors_x_test_tfidf.pkl      # TF-IDF test vectors
│   ├── labels_y_train.pkl            # Train labels
│   ├── labels_y_test.pkl             # Test labels
│   ├── embedding_matrix.pkl          # Word2Vec embedding matrix
│   ├── encoded_train_bert.pt         # BERT-encoded train features
│   └── encoded_test_bert.pt          # BERT-encoded test features
│
├── 01_Data_collection_scrap.ipynb    # Web scraping pipeline
├── 02_Preprocessing.ipynb            # Text cleaning & preprocessing
├── 03_Feature_extraction.ipynb       # TF-IDF, Word2Vec, BERT feature extraction
├── 04_Modeling_module.ipynb          # Model training, evaluation & comparison
│
├── main.ipynb                        # Launcher notebook (run the Streamlit app)
├── app.py                            # Streamlit web application
├── preprocessed.csv                  # Cleaned dataset (113,703 rows)
├── preprocessed.xlsx                 # Excel version of preprocessed data
└── README.md                         # This file
```

---

## Dataset

| Property | Value |
|---|---|
| Source | YouTube Comments API |
| Total comments | ~113,703 |
| Number of videos | 11 |
| Language filter | English only |
| Features | `comment`, `author`, `likes`, `published_at`, `video_title` |

### Videos Covered (Target Classes)
1. \$10,000 Every Day You Survive In The Wilderness
2. 100 Strongest Vs 100 Smartest Compete for \$5,000,000
3. 50 YouTubers Fight For \$1,000,000
4. AMP EXTREME OBSTACLE COURSE WITH MRBEAST
5. David Guetta - Hey Mama (Official Video) ft Nicki Minaj, Bebe Rexha & Afrojack
6. I Built 10 Schools Around The World
7. I Granted 100 Kids Their Biggest Wish!
8. I Spent 100 Hours Inside The Pyramids!
9. LAST TO LEAVE THE BOX FT AMP
10. Survive 100 Days Trapped In A Private Jet, Keep It
11. World's Deadliest Obstacle Course!

---

## NLP Pipeline

```
Raw Comments
     │
     ▼
01_Data_collection_scrap.ipynb
  └─ Scrape YouTube comments → Dataset/*.csv
     │
     ▼
02_Preprocessing.ipynb
  └─ Lowercase → Remove URLs, mentions, punctuation
  └─ Tokenization, stopword removal, lemmatization
  └─ Output: preprocessed.csv (final_cleaned_text column)
     │
     ▼
03_Feature_extraction.ipynb
  └─ TF-IDF Vectorizer (max_features=5000)
  └─ Word2Vec Embeddings
  └─ BERT Tokenization & Encoding
  └─ Output: features/*.pkl
     │
     ▼
04_Modeling_module.ipynb
  └─ Logistic Regression (primary model)
  └─ Train/test split: 80/20, random_state=42
  └─ Output: models/sentiment_logistic_regression_model.pkl
              models/tfidf_vectorizer.pkl
     │
     ▼
app.py + main.ipynb
  └─ Streamlit web UI for single & batch predictions
```

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | Logistic Regression |
| Vectorizer | TF-IDF (max 5000 features) |
| Train/Test Split | 80% / 20% |
| Random State | 42 |
| Number of Classes | 11 |
| Input | Raw YouTube comment text |
| Output | Predicted video title |

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- pip

### Install Dependencies

```bash
pip install streamlit joblib scikit-learn pandas
```

For running all notebooks (full pipeline):

```bash
pip install pandas scikit-learn nltk gensim transformers torch jupyter
```

---

## Running the Web App

### Option 1 — Terminal (Recommended)

```bash
cd 22CDS0446_NLP
streamlit run app.py
```

The app will open at **http://localhost:8501**

### Option 2 — Jupyter Notebook

Open `main.ipynb` in Jupyter and run all cells in order:
1. **Cell 1** — Install dependencies
2. **Cell 2** — Verify model files exist
3. **Cell 3** — (If needed) Regenerate vectorizer
4. **Cell 4** — Launch Streamlit app

---

## App Features

### Single Comment Prediction
- Enter any YouTube comment text
- Click **Predict** to classify it
- Shows predicted video category + probability scores for all 11 classes

### Batch Prediction (CSV)
- Upload a CSV file containing comments
- Select the column with comment text
- Get predictions for all rows
- Download results as CSV

---

## Preprocessing Steps

Applied in `02_Preprocessing.ipynb`:

1. **Lowercase** — Convert all text to lowercase
2. **URL removal** — Strip `http://`, `https://`, `www.*`
3. **Mention & hashtag removal** — Remove `@username` and `#hashtag`
4. **Punctuation removal** — Keep only alphabetic characters and spaces
5. **Whitespace normalization** — Collapse multiple spaces to single space
6. **Tokenization** — Split into tokens
7. **Stopword removal** — Remove common English stopwords (NLTK)
8. **Lemmatization** — Reduce words to base form

---

## Feature Extraction Methods

| Method | Description | Output |
|---|---|---|
| TF-IDF | Term Frequency-Inverse Document Frequency | Sparse matrix (90962 × 5000) |
| Word2Vec | Gensim Word2Vec embeddings | Embedding matrix |
| BERT | bert-base-uncased tokenizer + encoder | Dense tensors (max_length=128) |

> The final model uses **TF-IDF** features for classification.

---

## Report

The full project report is available as:
- `22CDS0446_NLP_Report_.pdf`
- `22CDS0446_NLP_Report_.docx`

---

## Author

| Field | Detail |
|---|---|
| Student ID | 22CDS0446 |
| Course | Natural Language Processing |
| Institution | VIT University |
