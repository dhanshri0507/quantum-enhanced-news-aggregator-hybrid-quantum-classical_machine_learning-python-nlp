import re
import string
import nltk
import pandas as pd
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from config import TFIDF_MAX_FEATURES, PCA_COMPONENTS

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = None
        self.pca = None
        self.is_fitted = False
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess a single text string"""
        if not text or pd.isna(text):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts"""
        return [self.preprocess_text(text) for text in texts]
    
    def fit_vectorizer(self, texts: List[str]):
        """Fit TF-IDF vectorizer on training data"""
        preprocessed_texts = self.preprocess_batch(texts)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(preprocessed_texts)
        
        # Fit PCA
        self.pca = PCA(n_components=PCA_COMPONENTS)
        self.pca.fit(tfidf_matrix.toarray())
        
        self.is_fitted = True
        print(f"Vectorizer fitted with {tfidf_matrix.shape[1]} features")
        print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
    
    def prepare_features(self, text: str) -> np.ndarray:
        """Convert text to feature vector ready for models"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before preparing features")
        
        # Preprocess text
        preprocessed_text = self.preprocess_text(text)
        
        # Transform to TF-IDF
        tfidf_vector = self.tfidf_vectorizer.transform([preprocessed_text])
        
        # Apply PCA
        pca_vector = self.pca.transform(tfidf_vector.toarray())
        
        return pca_vector.flatten()
    
    def prepare_features_batch(self, texts: List[str]) -> np.ndarray:
        """Convert batch of texts to feature vectors"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before preparing features")
        
        # Preprocess texts
        preprocessed_texts = self.preprocess_batch(texts)
        
        # Transform to TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.transform(preprocessed_texts)
        
        # Apply PCA
        pca_matrix = self.pca.transform(tfidf_matrix.toarray())
        
        return pca_matrix

def load_and_prepare_dataset(dataset_path: str) -> Tuple[List[str], List[float]]:
    """Load the news sentiment dataset and prepare it for training"""
    try:
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Use news_title as the main text feature
        texts = df['news_title'].fillna('').astype(str).tolist()
        
        # Convert sentiment to binary (assuming 1.0 is positive, 0.0 is negative)
        sentiments = df['sentiment'].fillna(0.0).astype(float).tolist()
        
        print(f"Loaded {len(texts)} samples from dataset")
        print(f"Sentiment distribution: {pd.Series(sentiments).value_counts().to_dict()}")
        
        return texts, sentiments
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return [], []

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Load dataset
    texts, sentiments = load_and_prepare_dataset("News_Sentiment_Dataset.csv")
    
    if texts:
        # Fit vectorizer
        preprocessor.fit_vectorizer(texts)
        
        # Test preprocessing
        test_text = "Breaking: Government announces new economic policy to boost growth!"
        features = preprocessor.prepare_features(test_text)
        print(f"Feature vector shape: {features.shape}")
        print(f"Feature vector: {features}")
