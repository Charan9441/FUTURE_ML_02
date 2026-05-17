import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

from preprocess import preprocess_text
from evaluate import evaluate_model

def train_pipeline(data_path='data/all_tickets_processed_improved_v3.csv', models_dir='models'):
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    print("Preprocessing text...")
    df['cleaned_text'] = df['Document'].apply(preprocess_text)
    
    X = df['cleaned_text']
    y_cat = df['Topic_group']
    
    # Split the data (80/20 train-test)
    X_train, X_test, y_cat_train, y_cat_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42
    )
    
    print("Extracting features (TF-IDF)...")
    # TF-IDF Vectorizer with max_features=5000, ngram_range=(1,2)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    # Fit on training data only, transform both
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Category Classifier
    print("Training Category Classifier...")
    cat_model = LogisticRegression(max_iter=1000, random_state=42)
    cat_model.fit(X_train_tfidf, y_cat_train)
    
    # Evaluate Category Classifier
    # Use unique classes found in the dataset
    cat_labels = list(cat_model.classes_)
    y_cat_pred = cat_model.predict(X_test_tfidf)
    evaluate_model(y_cat_test, y_cat_pred, cat_labels, "Category Classifier")
    
    # Save artifacts
    print(f"\nSaving models to {models_dir}/...")
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(vectorizer, os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
    joblib.dump(cat_model, os.path.join(models_dir, 'category_model.pkl'))
    
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    train_pipeline()
