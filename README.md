# Fake News Detection System

A machine learning-powered system that combines ML prediction with live news verification and fact-checking to identify potentially fake news articles.

## Features

- 🤖 Machine Learning Model
  - Trained on a large dataset of real and fake news articles
  - Uses Random Forest Classifier with TF-IDF vectorization
  - Achieves 99.74% accuracy on test data
  - Provides confidence scores and probability breakdowns

- 🔍 Live News Verification
  - Integration with NewsAPI for real-time news verification
  - Cross-references claims with trusted news sources
  - Provides related articles and sources

- ✅ Fact-Checking Integration
  - Google Fact Check API integration
  - Historical fact-check results
  - Multiple fact-checking sources

## Project Structure

```
fact_checker/
├── app/
│   ├── app.py          # Streamlit web application
│   └── utils.py        # Utility functions and API clients
├── model/
│   ├── train_model.py  # Model training script
│   ├── fake_news_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── model_info.pkl
└── data/
    ├── True.csv        # Real news dataset
    └── Fake.csv        # Fake news dataset
```

## Setup and Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fact_checker
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Train the model (if needed):
```bash
cd model
python train_model.py
```

5. Run the application:
```bash
cd app
streamlit run app.py
```

## API Keys Required

The application requires the following API keys:
- NewsAPI key (get from https://newsapi.org)
- Google Fact Check API key (get from Google Cloud Console)

## Model Performance

- Accuracy: 99.74%
- Cross-validation score: 99.78% (±0.10%)
- Training samples: 35,918
- Test samples: 8,980

## Usage

1. Launch the Streamlit app
2. Enter your API keys in the sidebar
3. Input a news headline or article text
4. Get comprehensive analysis including:
   - ML model prediction with confidence score
   - Related news articles from trusted sources
   - Fact-check results
   - Combined credibility assessment

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- NewsAPI for news verification
- Google Fact Check API for fact-checking 