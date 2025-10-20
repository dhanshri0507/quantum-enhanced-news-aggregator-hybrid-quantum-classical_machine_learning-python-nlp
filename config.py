# 
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GOOGLE_FC_KEY = os.getenv("GOOGLE_FC_KEY")
GNEWS_KEY = os.getenv("GNEWS_KEY")

# API Endpoints
NEWSAPI_URL = "https://newsapi.org/v2/top-headlines"
GNEWS_URL = "https://gnews.io/api/v4/search"
GOOGLE_FC_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# Model Configuration
TFIDF_MAX_FEATURES = 1000
PCA_COMPONENTS = 6  # Increased for better feature representation
QUANTUM_QUBITS = 6  # Increased for more expressivity

# Trust Score Weights
W1_FACTCHECK = 0.5
W2_SOURCE_REP = 0.3
W3_MULTI_SOURCE = 0.2

# Trust Badge Thresholds
HIGH_TRUST = 80
MEDIUM_TRUST = 50
