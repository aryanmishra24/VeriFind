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
        max_features=10000,  # Limit vocabulary size to top 10,000 most frequent terms
        ngram_range=(1, 2),  # Use unigrams (single words) and bigrams (two-word phrases)
        min_df=2,           # Ignore terms that appear in less than 2 documents (to remove very rare words)
        max_df=0.95,        # Ignore terms that appear in more than 95% of documents (to remove very common words)
        strip_accents='unicode', # Remove accents and perform other Unicode normalization
        lowercase=True      # Already handled by preprocess_text, but good for consistency
    )
    
    # Fit and transform training data, then transform test data
    print("\nVectorizing text data...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF feature matrix shape (training): {X_train_tfidf.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    # --- 9. Train multiple models and compare performance ---
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'), # Added solver for clarity
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), # n_jobs=-1 for parallel processing
        'Naive Bayes': MultinomialNB()
    }
    
    model_scores = {}
    trained_models = {}
    
    print("\n--- Training and evaluating models ---")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train_tfidf, y_train)
        trained_models[name] = model
        
        # Make predictions
        y_pred = model.predict(X_test_tfidf)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        model_scores[name] = accuracy
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"Classification Report for {name}:")
        print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))
    
    # Find best model
    best_model_name = max(model_scores, key=model_scores.get)
    best_model = trained_models[best_model_name]
    print(f"\nBest performing model: {best_model_name} with accuracy: {model_scores[best_model_name]:.4f}")

    # --- 10. Visualize model comparison ---
    plt.figure(figsize=(10, 6))
    models_list = list(model_scores.keys())
    accuracies = list(model_scores.values())
    
    bars = plt.bar(models_list, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title('Model Performance Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0.8, 1.0)  # Adjust based on your results
    
    # Add accuracy values on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                 f'{acc:.4f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show() # Display the plot

    # --- 11. Visualize confusion matrix for best model ---
    y_pred_best = best_model.predict(X_test_tfidf)
    cm = confusion_matrix(y_test, y_pred_best)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show() # Display the plot
    
    # Calculate and display additional metrics
    tn, fp, fn, tp = cm.ravel()
    # Handle potential division by zero if tp+fp or tp+fn is zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nDetailed Metrics for {best_model_name}:")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp} (Predicted FAKE, Actual REAL)")
    print(f"False Negatives (FN): {fn} (Predicted REAL, Actual FAKE)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # --- 12. Cross-validation for best model ---
    print(f"\nPerforming 5-fold cross-validation for {best_model_name}...")
    # Use the best model found and the TF-IDF transformed training data
    cv_scores = cross_val_score(best_model, X_train_tfidf, y_train, cv=5, scoring='accuracy', n_jobs=-1) # n_jobs=-1 for parallel processing
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Visualize CV scores
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 6), cv_scores, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
    plt.fill_between(range(1, 6), cv_scores.mean() - cv_scores.std(), 
                     cv_scores.mean() + cv_scores.std(), alpha=0.2, color='red')
    plt.title('Cross-Validation Scores')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show() # Display the plot

    # --- 13. Feature importance analysis ---
    if best_model_name == 'Random Forest':
        print("\nAnalyzing feature importance for Random Forest...")
        feature_names = vectorizer.get_feature_names_out()
        importances = best_model.feature_importances_
        
        # Get top 20 most important features
        indices = np.argsort(importances)[::-1][:20]
        
        plt.figure(figsize=(12, 8))
        plt.title('Top 20 Most Important Features (Random Forest)')
        plt.bar(range(20), importances[indices])
        plt.xticks(range(20), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.show() # Display the plot
        
        print("Top 10 most important features:")
        for i in range(10):
            print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    elif best_model_name == 'Logistic Regression':
        print("\nAnalyzing feature coefficients for Logistic Regression...")
        feature_names = vectorizer.get_feature_names_out()
        coefficients = best_model.coef_[0]
        
        # Get features most indicative of fake news (positive coefficients)
        fake_indices = np.argsort(coefficients)[::-1][:10]
        # Get features most indicative of real news (negative coefficients)
        real_indices = np.argsort(coefficients)[:10]
        
        print("\nTop 10 features indicating FAKE news (positive coefficients):")
        for i, idx in enumerate(fake_indices):
            print(f"{i+1}. {feature_names[idx]}: {coefficients[idx]:.4f}")
        
        print("\nTop 10 features indicating REAL news (negative coefficients):")
        for i, idx in enumerate(real_indices):
            print(f"{i+1}. {feature_names[idx]}: {coefficients[idx]:.4f}")
    else:
        print(f"\nFeature importance/coefficients analysis not implemented for {best_model_name}.")


    # --- 14. Save the trained model and vectorizer ---
    print("\n--- Saving model and vectorizer ---")
    
    # Define full paths for saving
    model_path = os.path.join(model_dir, 'fake_news_model.pkl')
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    info_path = os.path.join(model_dir, 'model_info.pkl')

    # Save the best model
    joblib.dump(best_model, model_path)
    print(f"Model saved as '{model_path}'")
    
    # Save the vectorizer
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Vectorizer saved as '{vectorizer_path}'")
    
    # Save model metadata
    model_info = {
        'model_name': best_model_name,
        'accuracy': model_scores[best_model_name],
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'features_count': X_train_tfidf.shape[1],
        'training_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'true_news_count': len(df_true),
        'fake_news_count': len(df_fake),
        'all_model_scores': model_scores
    }
    
    joblib.dump(model_info, info_path)
    print(f"Model information saved as '{info_path}'")
    print("\nTraining completed successfully!")
    print(f"\nFinal Model Summary:")
    print(f"- Best Model: {best_model_name}")
    print(f"- Test Accuracy: {model_scores[best_model_name]:.4f}")
    print(f"- Cross-validation Mean: {cv_scores.mean():.4f}")
    print(f"- Total Training Samples: {X_train.shape[0]:,}")
    print(f"- Total Test Samples: {X_test.shape[0]:,}")

    # --- 15. Test the saved model with sample predictions ---
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

