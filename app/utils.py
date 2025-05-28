"""
Utility functions for the Fake News Checker application
Handles ML prediction, NewsAPI integration, and Google Fact Check API
"""

import joblib
import re
import requests
import pandas as pd
from nltk.corpus import stopwords
import nltk
import os

# Download NLTK data if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class FakeNewsChecker:
    def __init__(self):
        """Initialize the checker with pre-trained model and vectorizer"""
        self.model = None
        self.vectorizer = None
        self.model_info = None
        self.load_model()
    
    def load_model(self):
        """Load the pre-trained model and vectorizer"""
        try:
            # Get the absolute path to the model directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(base_dir, 'model')
            
            self.model = joblib.load(os.path.join(model_dir, 'fake_news_model.pkl'))
            self.vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
            self.model_info = joblib.load(os.path.join(model_dir, 'model_info.pkl'))
            print("Model loaded successfully!")
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            print("Please ensure the model files exist in the 'model' directory")
            print("Run the training notebook first to generate the model files")
    
    def preprocess_text(self, text):
        """
        Preprocess text for prediction
        Same preprocessing as used during model training
        """
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def predict_headline(self, headline):
        """
        Predict if a headline is fake or real
        Returns prediction, confidence, and probabilities
        """
        if self.model is None or self.vectorizer is None:
            return {
                'prediction': 'Unknown',
                'confidence': 0,
                'error': 'Model not loaded'
            }
        
        try:
            # Preprocess the headline
            processed_text = self.preprocess_text(headline)
            
            if not processed_text:
                return {
                    'prediction': 'Unknown',
                    'confidence': 0,
                    'error': 'Invalid input text'
                }
            
            # Vectorize the text
            vectorized = self.vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = self.model.predict(vectorized)[0]
            probabilities = self.model.predict_proba(vectorized)[0]
            
            # Format results
            result = "FAKE" if prediction == 1 else "REAL"
            confidence = max(probabilities) * 100
            
            return {
                'prediction': result,
                'confidence': round(confidence, 2),
                'fake_probability': round(probabilities[1] * 100, 2),
                'real_probability': round(probabilities[0] * 100, 2),
                'processed_text': processed_text
            }
            
        except Exception as e:
            return {
                'prediction': 'Unknown',
                'confidence': 0,
                'error': f'Prediction error: {str(e)}'
            }

class NewsAPIClient:
    def __init__(self, api_key):
        """Initialize NewsAPI client with API key"""
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
    
    def search_news(self, query, max_results=5):
        """
        Search for news articles related to the query
        Returns list of articles with title, source, url, and description
        """
        if not self.api_key:
            return {
                'articles': [],
                'error': 'NewsAPI key not provided'
            }
        
        try:
            # Clean and prepare search query
            clean_query = re.sub(r'[^\w\s]', '', query)
            
            # NewsAPI everything endpoint for comprehensive search
            url = f"{self.base_url}/everything"
            params = {
                'q': clean_query,
                'apiKey': self.api_key,
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': max_results,
                'domains': 'reuters.com,bbc.com,cnn.com,apnews.com,npr.org,wsj.com,nytimes.com'  # Trusted sources
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            articles = []
            if data.get('status') == 'ok' and data.get('articles'):
                for article in data['articles'][:max_results]:
                    articles.append({
                        'title': article.get('title', 'No title'),
                        'source': article.get('source', {}).get('name', 'Unknown source'),
                        'url': article.get('url', ''),
                        'description': article.get('description', 'No description'),
                        'published_at': article.get('publishedAt', ''),
                        'url_to_image': article.get('urlToImage', '')
                    })
            
            return {
                'articles': articles,
                'total_results': data.get('totalResults', 0),
                'query_used': clean_query
            }
            
        except requests.RequestException as e:
            return {
                'articles': [],
                'error': f'NewsAPI request failed: {str(e)}'
            }
        except Exception as e:
            return {
                'articles': [],
                'error': f'NewsAPI error: {str(e)}'
            }

class FactCheckAPI:
    def __init__(self, api_key):
        """Initialize Google Fact Check API client"""
        self.api_key = api_key
        self.base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    
    def search_fact_checks(self, query, max_results=5):
        """
        Search for fact-checks related to the query
        Returns list of fact-check results
        """
        if not self.api_key:
            return {
                'fact_checks': [],
                'error': 'Google Fact Check API key not provided'
            }
        
        try:
            # Clean query for fact-check search
            clean_query = re.sub(r'[^\w\s]', '', query)
            
            params = {
                'query': clean_query,
                'key': self.api_key,
                'languageCode': 'en',
                'maxAgeDays': 365,  # Look for fact-checks up to 1 year old
                'pageSize': max_results
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            fact_checks = []
            if data.get('claims'):
                for claim in data['claims'][:max_results]:
                    # Extract relevant information
                    claim_text = claim.get('text', 'No claim text')
                    claim_review = claim.get('claimReview', [{}])[0]
                    
                    fact_checks.append({
                        'claim': claim_text,
                        'rating': claim_review.get('textualRating', 'No rating'),
                        'reviewer': claim_review.get('publisher', {}).get('name', 'Unknown reviewer'),
                        'url': claim_review.get('url', ''),
                        'title': claim_review.get('title', 'No title'),
                        'review_date': claim_review.get('reviewDate', ''),
                        'claim_date': claim.get('claimDate', '')
                    })
            
            return {
                'fact_checks': fact_checks,
                'query_used': clean_query
            }
            
        except requests.RequestException as e:
            return {
                'fact_checks': [],
                'error': f'Fact Check API request failed: {str(e)}'
            }
        except Exception as e:
            return {
                'fact_checks': [],
                'error': f'Fact Check API error: {str(e)}'
            }

def get_combined_verdict(ml_prediction, news_articles, fact_checks):
    """
    Combine ML prediction with news verification and fact-checks
    to provide a comprehensive verdict
    """
    verdict = {
        'overall_assessment': 'Unknown',
        'confidence_level': 'Low',
        'reasoning': [],
        'credibility_score': 0
    }
    
    # Start with ML prediction
    ml_confidence = ml_prediction.get('confidence', 0)
    ml_result = ml_prediction.get('prediction', 'Unknown')
    
    credibility_score = 0
    reasoning = []
    
    # Factor 1: ML Model prediction (40% weight)
    if ml_result == 'FAKE':
        credibility_score -= (ml_confidence * 0.4) / 100
        reasoning.append(f"ML model predicts FAKE with {ml_confidence}% confidence")
    elif ml_result == 'REAL':
        credibility_score += (ml_confidence * 0.4) / 100
        reasoning.append(f"ML model predicts REAL with {ml_confidence}% confidence")
    
    # Factor 2: News articles verification (30% weight)
    if news_articles and len(news_articles) > 0:
        trusted_sources = ['reuters', 'bbc', 'cnn', 'ap news', 'npr', 'wsj', 'new york times']
        trusted_count = sum(1 for article in news_articles 
                          if any(source in article['source'].lower() for source in trusted_sources))
        
        news_score = min(len(news_articles) * 0.1, 0.3)  # Max 30% boost
        credibility_score += news_score
        
        if trusted_count > 0:
            reasoning.append(f"Found {len(news_articles)} related articles, {trusted_count} from trusted sources")
        else:
            reasoning.append(f"Found {len(news_articles)} related articles")
    else:
        reasoning.append("No related news articles found")
    
    # Factor 3: Fact-check results (30% weight)
    if fact_checks and len(fact_checks) > 0:
        false_ratings = ['false', 'mostly false', 'pants on fire', 'fake']
        true_ratings = ['true', 'mostly true', 'correct']
        
        false_count = sum(1 for fc in fact_checks 
                         if any(rating in fc['rating'].lower() for rating in false_ratings))
        true_count = sum(1 for fc in fact_checks 
                        if any(rating in fc['rating'].lower() for rating in true_ratings))
        
        if false_count > true_count:
            credibility_score -= 0.3
            reasoning.append(f"{false_count} fact-checks indicate false claims")
        elif true_count > false_count:
            credibility_score += 0.3
            reasoning.append(f"{true_count} fact-checks support the claims")
        else:
            reasoning.append(f"Found {len(fact_checks)} fact-checks with mixed ratings")
    else:
        reasoning.append("No related fact-checks found")
    
    # Normalize credibility score to 0-100 scale
    credibility_score = max(0, min(100, (credibility_score + 1) * 50))
    
    # Determine overall assessment
    if credibility_score >= 75:
        overall_assessment = "Likely REAL"
        confidence_level = "High"
    elif credibility_score >= 60:
        overall_assessment = "Possibly REAL"
        confidence_level = "Medium"
    elif credibility_score >= 40:
        overall_assessment = "Uncertain"
        confidence_level = "Low"
    elif credibility_score >= 25:
        overall_assessment = "Possibly FAKE"
        confidence_level = "Medium"
    else:
        overall_assessment = "Likely FAKE"
        confidence_level = "High"
    
    verdict.update({
        'overall_assessment': overall_assessment,
        'confidence_level': confidence_level,
        'reasoning': reasoning,
        'credibility_score': round(credibility_score, 1)
    })
    
    return verdict