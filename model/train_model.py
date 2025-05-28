# train_model.py

# This script trains a machine learning model to classify news headlines as fake or real
# using True.csv and Fake.csv datasets.
# It performs data loading, preprocessing, model training, evaluation, visualization,
# and saves the trained model and vectorizer.

import pandas as pd
import numpy as np
import re
import string
import joblib
import os # Import os for path manipulation

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline # Not directly used in the final script structure, but good to keep if pipelines were considered.

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Download NLTK data ---
# This step is crucial for text preprocessing. Ensure you have an internet connection.
print("Downloading NLTK data (stopwords, punkt)...")
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
print("NLTK data download complete.")

# --- 2. Text preprocessing function ---
def preprocess_text(text):
    """
    Preprocess text by:
    - Converting to lowercase
    - Removing punctuation and special characters
    - Removing stopwords
    - Tokenizing (implicitly by splitting and joining)
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    # This regex keeps only alphabetic characters and whitespace
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace (e.g., multiple spaces between words)
    text = ' '.join(text.split())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    # Filter out stopwords and words shorter than 3 characters (e.g., 'a', 'an', 'is')
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

# --- Main script execution ---
if __name__ == "__main__":
    # Define data and model directories
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data'))
    model_dir = script_dir  # Save models directly in the model directory

    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # --- 3. Load the datasets ---
    print("Loading datasets...")
    
    try:
        # Load True news dataset
        true_csv_path = os.path.join(data_dir, 'True.csv')
        df_true = pd.read_csv(true_csv_path)
        df_true['label'] = 'REAL'  # Add label for real news
        print(f"True news dataset shape: {df_true.shape}")
    
        # Load Fake news dataset
        fake_csv_path = os.path.join(data_dir, 'Fake.csv')
        df_fake = pd.read_csv(fake_csv_path)
        df_fake['label'] = 'FAKE'  # Add label for fake news
        print(f"Fake news dataset shape: {df_fake.shape}")
    
        # Combine both datasets
        df = pd.concat([df_true, df_fake], ignore_index=True)
        print(f"Combined dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    
        # Remove the subject column as requested
        if 'subject' in df.columns:
            df = df.drop('subject', axis=1)
            print("Subject column removed")
    
        print(f"Final columns: {df.columns.tolist()}")
        print("\nFirst few rows:")
        print(df.head())

    except FileNotFoundError as e:
        print(f"Error: Dataset file not found. Please ensure 'True.csv' and 'Fake.csv' are in the '{data_dir}' directory.")
        print(f"Details: {e}")
        exit() # Exit if data files are not found

    # --- 4. Explore the dataset ---
    print("\n--- Dataset Info ---")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nLabel distribution:")
    print(df['label'].value_counts())
    
    # Visualize label distribution
    plt.figure(figsize=(8, 6))
    df['label'].value_counts().plot(kind='bar')
    plt.title('Distribution of Real vs Fake News')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show() # Display the plot

    # --- 5. Data cleaning and preprocessing ---
    print("\n--- Preprocessing data ---")
    
    # Handle missing values by filling with empty strings
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    
    # Combine title and text for better prediction (use more context)
    df['combined_text'] = df['title'] + ' ' + df['text']
    
    # Preprocess the combined text
    print("Applying text preprocessing...")
    df['processed_text'] = df['combined_text'].apply(preprocess_text)
    
    # Remove rows where processed_text is empty after cleaning
    df = df[df['processed_text'].str.len() > 0]
    
    print(f"Dataset shape after cleaning: {df.shape}")
    print("\nLabel distribution after cleaning:")
    print(df['label'].value_counts())
    print("\nSample processed text (first 200 chars):")
    if not df.empty:
        print(df['processed_text'].iloc[0][:200] + "...")
    else:
        print("No processed text available after cleaning.")

    # --- 6. Prepare features and labels ---
    X = df['processed_text']
    y = df['label']
    
    # Convert labels to binary (0 for REAL, 1 for FAKE)
    y = (y == 'FAKE').astype(int)
    
    print(f"\nFeature shape: {X.shape}")
    print(f"Label distribution (0=REAL, 1=FAKE):\n{y.value_counts()}")
    print(f"Percentage of fake news: {y.mean() * 100:.1f}%")
    print(f"Percentage of real news: {(1 - y.mean()) * 100:.1f}%")

    # --- 7. Split the data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Training set fake news percentage: {y_train.mean() * 100:.1f}%")
    print(f"Test set fake news percentage: {y_test.mean() * 100:.1f}%")

    # --- 8. Initialize TF-IDF Vectorizer ---
    vectorizer = TfidfVectorizer(
        max_features=2000,  # Further reduced from 5000 to 2000
        ngram_range=(1, 1),  # Only use unigrams to reduce size
        min_df=5,           # Increased from 3 to 5 to remove more rare terms
        max_df=0.85,        # Further reduced from 0.90 to 0.85
        strip_accents='unicode',
        lowercase=True,
        dtype=np.float32,   # Use float32 for memory efficiency
        sublinear_tf=True   # Apply sublinear scaling to reduce feature weights
    )
    
    # Fit and transform training data, then transform test data
    print("\nVectorizing text data...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF feature matrix shape (training): {X_train_tfidf.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    # --- 9. Train and evaluate models ---
    print("\n--- Training and evaluating models ---")
    
    # Initialize models - using only lighter models
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0,
            max_iter=1000,
            n_jobs=-1,
            random_state=42,
            solver='liblinear'  # More memory efficient solver
        ),
        'Multinomial Naive Bayes': MultinomialNB()
    }
    
    # Train and evaluate each model
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_tfidf, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Update best model
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_model_name = name
    
    print(f"\nBest model: {best_model_name} with accuracy: {best_score:.4f}")
    
    # --- 10. Save the best model and vectorizer ---
    print("\n--- Saving model and vectorizer ---")
    
    # Save the vectorizer with maximum compression
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    joblib.dump(vectorizer, vectorizer_path, compress=9)  # Maximum compression
    print(f"Vectorizer saved to {vectorizer_path}")
    
    # Save the best model with maximum compression
    model_path = os.path.join(model_dir, 'fake_news_model.pkl')
    joblib.dump(best_model, model_path, compress=9)  # Maximum compression
    print(f"Model saved to {model_path}")
    
    # Save model information
    model_info = {
        'model_name': best_model_name,
        'accuracy': best_score,
        'features_count': len(vectorizer.vocabulary_),
        'training_samples': X_train.shape[0],
        'vectorizer_params': vectorizer.get_params()
    }
    
    info_path = os.path.join(model_dir, 'model_info.pkl')
    joblib.dump(model_info, info_path)
    print(f"Model info saved to {info_path}")
    
    print("\nTraining complete!")

    # --- 11. Test the saved model with sample predictions ---
    print("\n--- Testing saved model ---")
    
    try:
        # Load the saved model and vectorizer
        loaded_model = joblib.load(model_path)
        loaded_vectorizer = joblib.load(vectorizer_path)
        loaded_info = joblib.load(info_path)
        
        print(f"Loaded model: {loaded_info['model_name']}")
        print(f"Model accuracy: {loaded_info['accuracy']:.4f}")
        
        # Test with sample headlines
        test_headlines = [
            "Scientists discover breakthrough cure for cancer in major clinical trial",
            "Local man wins lottery twice in same week using lucky numbers",
            "Government announces new tax policy changes effective next year",
            "Aliens spotted landing in downtown area witnessed by hundreds",
            "Stock market reaches all-time high amid economic recovery",
            "Celebrity couple announces divorce after secret wedding last month",
            "New study shows coffee consumption linked to longer lifespan",
            "Breaking: President makes surprise announcement about healthcare reform"
        ]
        
        print("\nSample predictions:")
        print("=" * 80)
        for i, headline in enumerate(test_headlines, 1):
            # Preprocess the headline using the same function
            processed = preprocess_text(headline)
            
            # Vectorize the processed text using the loaded vectorizer
            vectorized = loaded_vectorizer.transform([processed])
            
            # Predict the label
            prediction = loaded_model.predict(vectorized)[0]
            # Get prediction probabilities for both classes
            probability = loaded_model.predict_proba(vectorized)[0]
            
            # Determine the result string
            result = "FAKE" if prediction == 1 else "REAL"
            # Calculate confidence as the maximum probability
            confidence = max(probability) * 100
            
            print(f"{i}. Headline: {headline}")
            print(f"   Prediction: {result} (Confidence: {confidence:.1f}%)")
            print(f"   Probabilities: Real={probability[0]:.3f}, Fake={probability[1]:.3f}")
            print("-" * 80)

    except FileNotFoundError as e:
        print(f"Error: Saved model or vectorizer files not found in '{model_dir}'.")
        print(f"Please ensure the training phase completed successfully. Details: {e}")
    except Exception as e:
        print(f"An error occurred during model testing: {e}")

