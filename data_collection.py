import requests
import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from config import NEWSAPI_KEY, GNEWS_KEY, NEWSAPI_URL, GNEWS_URL

class NewsCollector:
    def __init__(self):
        self.newsapi_key = NEWSAPI_KEY
        self.gnews_key = GNEWS_KEY
        
    def find_crossref_sources_for_title(self, title: str, max_results: int = 10) -> List[str]:
        """Find similar headlines to a single title using GNews and return source names."""
        if not title:
            return []
        # Build a lightweight query from keywords
        keywords = self._extract_keywords(title)
        query = " ".join(keywords[:3]) if keywords else title
        candidates = self.fetch_gnews_headlines(query, max_results=max_results)
        similar_sources: List[str] = []
        for art in candidates:
            if self._is_similar_headline(title, art.get("title")):
                if art.get("source"):
                    similar_sources.append(art["source"])
        return list(set(similar_sources))

    def fetch_newsapi_headlines(self, query: str = "economy OR policy", 
                            language: str = "en", page_size: int = 50) -> List[Dict]:
        """Fetch headlines from NewsAPI"""
        try:
            if not self.newsapi_key:
                print("NEWSAPI_KEY not set. Skipping NewsAPI and trying GNews fallback.")
                return self.fetch_gnews_headlines(query=query, max_results=min(page_size, 10))

            params = {
                "q": query,
                "language": language,
                "pageSize": page_size,
                # NewsAPI top-headlines requires one of: country, category, or sources
                # We default to U.S. general headlines to avoid empty responses
                "country": "us",
                "category": "general",
                "apiKey": self.newsapi_key
            }
            
            response = requests.get(NEWSAPI_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            # Handle NewsAPI logical errors that still return HTTP 200
            if data.get("status") != "ok":
                message = data.get("message", "Unknown error from NewsAPI")
                print(f"NewsAPI returned error status: {message}")
                # Fallback to GNews
                return self.fetch_gnews_headlines(query=query, max_results=min(page_size, 10))
            headlines = []
            
            for article in data.get("articles", []):
                headline = {
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "source": article.get("source", {}).get("name"),
                    "publishedAt": article.get("publishedAt"),
                    "url": article.get("url"),
                    "api_source": "newsapi"
                }
                headlines.append(headline)
                
            # If NewsAPI returned no items, try GNews as fallback
            if not headlines:
                print("NewsAPI returned zero articles. Trying GNews fallback...")
                return self.fetch_gnews_headlines(query=query, max_results=min(page_size, 10))
            
            return headlines
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching from NewsAPI: {e}")
            # Fallback to GNews if network error specific to NewsAPI
            return self.fetch_gnews_headlines(query=query, max_results=min(page_size, 10))
    
    def fetch_gnews_headlines(self, query: str, max_results: int = 20) -> List[Dict]:
        """Fetch headlines from GNews for cross-validation"""
        try:
            params = {
                "q": query,
                "lang": "en",
                "max": max_results,
                "token": self.gnews_key
            }
            
            response = requests.get(GNEWS_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            headlines = []
            
            for article in data.get("articles", []):
                headline = {
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "source": article.get("source", {}).get("name"),
                    "publishedAt": article.get("publishedAt"),
                    "url": article.get("url"),
                    "api_source": "gnews"
                }
                headlines.append(headline)
                
            return headlines
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching from GNews: {e}")
            return []
    
    def cross_validate_headlines(self, main_headlines: List[Dict]) -> Dict[str, List[str]]:
        """Cross-validate headlines across multiple sources"""
        crossrefs = {}
        
        for headline in main_headlines:
            title = headline["title"]
            if not title:
                continue
                
            # Extract keywords from title
            keywords = self._extract_keywords(title)
            query = " ".join(keywords[:3])  # Use top 3 keywords
            
            # Fetch from GNews
            gnews_results = self.fetch_gnews_headlines(query, max_results=10)
            
            # Find similar headlines
            similar_sources = []
            for gnews_article in gnews_results:
                if self._is_similar_headline(title, gnews_article["title"]):
                    similar_sources.append(gnews_article["source"])
            
            crossrefs[title] = list(set(similar_sources))
            
        return crossrefs
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        if not text:
            return []
        
        # Simple keyword extraction (remove common words)
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        words = text.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:5]  # Return top 5 keywords
    
    def _is_similar_headline(self, headline1: str, headline2: str, threshold: float = 0.6) -> bool:
        """Check if two headlines are similar using simple word overlap"""
        if not headline1 or not headline2:
            return False
            
        words1 = set(headline1.lower().split())
        words2 = set(headline2.lower().split())
        
        if not words1 or not words2:
            return False
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union) if union else 0
        return similarity >= threshold

# Example usage
if __name__ == "__main__":
    collector = NewsCollector()
    
    # Fetch headlines from NewsAPI
    headlines = collector.fetch_newsapi_headlines()
    print(f"Fetched {len(headlines)} headlines from NewsAPI")
    
    # Cross-validate with GNews
    crossrefs = collector.cross_validate_headlines(headlines[:5])  # Test with first 5
    print("Cross-validation results:")
    for title, sources in crossrefs.items():
        print(f"'{title}': {sources}")
