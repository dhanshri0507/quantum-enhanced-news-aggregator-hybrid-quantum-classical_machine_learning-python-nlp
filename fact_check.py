import requests  # type: ignore
import json
from typing import Dict, List, Optional, Tuple
from config import GOOGLE_FC_KEY, GOOGLE_FC_URL

class FactChecker:
    def __init__(self):
        self.google_fc_key = GOOGLE_FC_KEY
        
    def factcheck_google(self, headline: str) -> float:
        """Query Google Fact Check API for headline verification"""
        try:
            print(f"🔍 Fact-checking: '{headline[:50]}...'")
            
            # Check if API key is available
            if not self.google_fc_key:
                print("⚠️ Google Fact Check API key not found, returning neutral score")
                return 0.5
                
            params = {
                "query": headline,
                "languageCode": "en-US",
                "key": self.google_fc_key
            }
            
            response = requests.get(GOOGLE_FC_URL, params=params, timeout=10)
            print(f"📡 Google FC response status: {response.status_code}")
            
            if response.status_code == 403:
                print("❌ Google Fact Check API: Access forbidden - check API key and billing")
                return 0.5
            elif response.status_code == 429:
                print("❌ Google Fact Check API: Rate limit exceeded")
                return 0.5
                
            response.raise_for_status()
            
            data = response.json()
            print(f"📊 Google FC response: {data}")
            
            # Extract fact-check rating
            claims = data.get("claims", [])
            if not claims:
                print("❌ No claims found in Google FC response")
                return 0.5  # Neutral score if no claims found
            
            # Get the first claim's rating
            claim = claims[0]
            claim_reviews = claim.get("claimReview", [])
            
            if not claim_reviews:
                print("❌ No claim reviews found")
                return 0.5  # Neutral score if no reviews
            
            # Extract textual rating
            textual_rating = claim_reviews[0].get("textualRating", "Unknown")
            publisher = claim_reviews[0].get("publisher", {}).get("name", "Unknown")
            
            # Map rating to score - Enhanced mapping for Google Fact Check API
            rating_scores = {
                "True": 1.0,
                "Correct": 1.0,
                "Mostly True": 0.8,
                "Mostly Correct": 0.8,
                "Half True": 0.5,
                "Mixed": 0.5,
                "Unproven": 0.5,
                "False": 0.0,
                "Incorrect": 0.0,
                "This is false": 0.0,
                "This is true": 1.0,
                "This is mostly true": 0.8,
                "This is mostly false": 0.2,
                "Misleading": 0.2,
                "Outdated": 0.3,
                "Pants on Fire": 0.0,
                "Four Pinocchios": 0.0,
                "Three Pinocchios": 0.2,
                "Two Pinocchios": 0.4,
                "One Pinocchio": 0.6,
                "No Pinocchios": 1.0,
                "Geppetto Checkmark": 1.0
            }
            
            score = rating_scores.get(textual_rating, 0.5)
            print(f"✅ Google Fact Check: '{headline[:30]}...' -> {textual_rating} by {publisher} (score: {score})")
            
            return score
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error querying Google Fact Check API: {e}")
            return 0.5  # Neutral score on error
        except Exception as e:
            print(f"❌ Error processing fact check response: {e}")
            return 0.5

class SourceReputation:
    def __init__(self):
        # Source reputation database (simplified)
        self.source_ratings = {
            # High credibility sources
            "BBC": 0.95,
            "Reuters": 0.93,
            "Associated Press": 0.92,
            "The New York Times": 0.90,
            "The Washington Post": 0.89,
            "The Guardian": 0.88,
            "CNN": 0.85,
            "NPR": 0.87,
            "PBS": 0.86,
            "The Wall Street Journal": 0.91,
            "Financial Times": 0.90,
            "Bloomberg": 0.88,
            "The Economist": 0.89,
            "Time": 0.85,
            "Newsweek": 0.84,
            "USA Today": 0.82,
            "Los Angeles Times": 0.83,
            "Chicago Tribune": 0.81,
            "Boston Globe": 0.82,
            "Miami Herald": 0.80,
            "NASA": 0.95,  
            "National Aeronautics and Space Administration": 0.95,
            
            # Medium credibility sources
            "Fox News": 0.65,
            "MSNBC": 0.70,
            "CBS News": 0.78,
            "ABC News": 0.77,
            "NBC News": 0.79,
            "Politico": 0.75,
            "The Hill": 0.73,
            "Axios": 0.76,
            "Vox": 0.72,
            "BuzzFeed News": 0.68,
            "HuffPost": 0.66,
            "Business Insider": 0.74,
            "Forbes": 0.77,
            "CNBC": 0.75,
            "MarketWatch": 0.73,
            
            # Lower credibility sources
            "Breitbart": 0.35,
            "Infowars": 0.15,
            "Natural News": 0.20,
            "The Gateway Pundit": 0.25,
            "Daily Caller": 0.40,
            "The Blaze": 0.45,
            "One America News": 0.30,
            "Newsmax": 0.50,
            
            # Default for unknown sources
            "Unknown": 0.40,

            # Entertainment / Tech / Sports commonly seen
            "Variety": 0.78,
            "The Hollywood Reporter": 0.77,
            "Deadline": 0.78,
            "People": 0.70,
            "ESPN": 0.82,
            "Bleacher Report": 0.70,
            "The Verge": 0.80,
            "Engadget": 0.78,
            "Gizmodo": 0.68,
            "TechCrunch": 0.80,
            "Ars Technica": 0.82,
            "CNET": 0.74,
            "Wired": 0.82,
            "Al Jazeera": 0.86,
            "Times of India": 0.70,
            "The Hindu": 0.78,
            "NDTV": 0.72,
            
            # Scientific and Academic Sources
            "Nature": 0.95,
            "Science": 0.95,
            "Scientific American": 0.90,
            "New Scientist": 0.88,
            "National Geographic": 0.89,
            "Smithsonian": 0.87,
            "MIT Technology Review": 0.88,
            "Harvard Medical School": 0.94,
            "Johns Hopkins": 0.93,
            "Mayo Clinic": 0.92,
            "WebMD": 0.80,
            "Healthline": 0.78,
            
            # Government and Official Sources
            "White House": 0.90,
            "CDC": 0.94,
            "WHO": 0.92,
            "FDA": 0.91,
            "NIH": 0.93,
            "NOAA": 0.92,
            "USGS": 0.91,
            "FBI": 0.88,
            "CIA": 0.85,
            "Pentagon": 0.87,
            "Congress": 0.80,
            "Supreme Court": 0.89
        }
    
    def _normalize(self, source_name: str) -> str:
        s = source_name or "Unknown"
        s = s.strip()
        # Remove common suffixes / adornments
        lowers = s.lower()
        for token in [".com", ".co.uk", " edition", " - us", " (uk)", " (us)"]:
            if token in lowers:
                lowers = lowers.replace(token, "")
        # Title-case back
        return " ".join(part.capitalize() for part in lowers.split()) or "Unknown"

    def get_source_reputation(self, source_name: str) -> float:
        """Get reputation score for a news source"""
        if not source_name:
            return self.source_ratings["Unknown"]
        # Normalize variations
        source_clean = self._normalize(source_name)
        
        # Direct lookup
        if source_clean in self.source_ratings:
            return self.source_ratings[source_clean]
        
        # Partial matching for variations
        for known_source, rating in self.source_ratings.items():
            if known_source.lower() in source_clean.lower() or source_clean.lower() in known_source.lower():
                return rating
        
        # Return default for unknown sources
        return self.source_ratings["Unknown"]

class MultiSourceVerifier:
    def __init__(self):
        self.reputation_checker = SourceReputation()
    
    def multi_source_match(self, headline: str, crossref_sources: List[str]) -> float:
        """Calculate multi-source verification score"""
        if not crossref_sources:
            return 0.0
        
        # Count reputable sources (score >= 0.7)
        reputable_count = 0
        for source in crossref_sources:
            if self.reputation_checker.get_source_reputation(source) >= 0.7:
                reputable_count += 1
        
        # Calculate score: min(1.0, 0.15 * k) where k is number of reputable sources
        score = min(1.0, 0.15 * reputable_count)
        
        print(f"Multi-source verification: {reputable_count} reputable sources -> score: {score}")
        return score

class TrustScoreCalculator:
    def __init__(self):
        self.fact_checker = FactChecker()
        self.reputation_checker = SourceReputation()
        self.multi_source_verifier = MultiSourceVerifier()
    
    def _best_factcheck_score(self, headline: str) -> float:
        """Try multiple query variants and return the best fact-check score."""
        candidates = [headline]
        # Strip quotes / punctuation and shorten
        import re
        base = re.sub(r"[\-\:\|\"\'\(\)\[\]\!\?]", " ", headline)
        base = re.sub(r"\s+", " ", base).strip()
        parts = base.split()
        if len(parts) > 4:
            candidates.append(" ".join(parts[:6]))
        # Keywords subset
        keywords = [w for w in parts if w[0].isalpha() and len(w) > 3]
        if keywords:
            candidates.append(" ".join(keywords[:5]))
        
        best = 0.5
        for q in candidates:
            try:
                best = max(best, self.fact_checker.factcheck_google(q))
            except Exception:
                pass
        return best
    
    def compute_trust_score(self, fc_score: float, rep_score: float, multi_score: float,
                        w1: float = 0.3, w2: float = 0.5, w3: float = 0.2) -> int:
        """Compute final trust score using weighted combination with improved weights"""
        # Boost score if we have high source reputation and multi-source verification
        if rep_score >= 0.9 and multi_score > 0.4:
            # Apply a significant bonus for highly reputable sources with strong multi-source verification
            bonus = 0.15
        elif rep_score >= 0.8 and multi_score > 0.3:
            # Apply a moderate bonus for good sources with multi-source verification
            bonus = 0.1
        else:
            bonus = 0.0
            
        score = w1 * fc_score + w2 * rep_score + w3 * multi_score + bonus
        return min(100, round(100 * score))  # Cap at 100
    
    def get_trust_badge(self, trust_score: int) -> Tuple[str, str]:
        """Get trust badge based on score"""
        if trust_score >= 80:
            return "High Trust", "🟢"
        elif trust_score >= 50:
            return "Medium Trust", "🟡"
        else:
            return "Low Trust", "🔴"
    
    def calculate_complete_trust_score(self, headline: str, source: str, 
                                    crossref_sources: List[str] = None) -> Dict:
        """Calculate complete trust score for a headline"""
        # Fact-check score (best-of multiple queries)
        fc_score = self._best_factcheck_score(headline)
        
        # Source reputation score
        rep_score = self.reputation_checker.get_source_reputation(source)
        
        # Multi-source verification score (improved calculation)
        multi_score = 0.0
        if crossref_sources:
            reputable_count = 0
            total_weight = 0.0
            for src in crossref_sources:
                r = self.reputation_checker.get_source_reputation(src)
                if r >= 0.7:
                    reputable_count += 1
                    total_weight += r
            
            if reputable_count > 0:
                # Improved calculation: each reputable source contributes more
                avg_reputation = total_weight / reputable_count
                # Base score from number of sources, scaled by average reputation
                base_score = min(0.6, 0.15 * reputable_count)  # Up to 0.6 for 4+ sources
                reputation_multiplier = avg_reputation  # Scale by reputation quality
                multi_score = min(1.0, base_score * reputation_multiplier)
        
        # Compute final trust score
        trust_score = self.compute_trust_score(fc_score, rep_score, multi_score)
        
        # Get trust badge
        badge_name, badge_icon = self.get_trust_badge(trust_score)
        
        return {
            "fact_check_score": fc_score,
            "source_reputation_score": rep_score,
            "multi_source_score": multi_score,
            "trust_score": trust_score,
            "trust_badge": badge_name,
            "trust_icon": badge_icon,
            "verification_details": {
                "fact_checked": fc_score > 0.5,
                "reputable_source": rep_score >= 0.7,
                "multi_source_verified": multi_score > 0.0
            }
        }

# Example usage
if __name__ == "__main__":
    trust_calculator = TrustScoreCalculator()
    
    # Test cases with different scenarios
    test_cases = [
        {
            "headline": "NASA successfully launches James Webb Space Telescope",
            "source": "NASA",
            "crossrefs": ["BBC", "Reuters", "Associated Press", "The New York Times"],
            "description": "High trust scenario - reputable source with multiple verifications"
        },
        {
            "headline": "Government announces new economic stimulus package",
            "source": "Reuters",
            "crossrefs": ["BBC", "Associated Press", "The New York Times"],
            "description": "Medium trust scenario - good source but may have fact-check issues"
        },
        {
            "headline": "Breaking: Major scientific breakthrough announced",
            "source": "The Guardian",
            "crossrefs": ["Scientific American", "Nature", "Science"],
            "description": "High trust scenario - reputable source with scientific backing"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}: {test_case['description']}")
        print(f"{'='*60}")
        
        trust_result = trust_calculator.calculate_complete_trust_score(
            test_case["headline"], test_case["source"], test_case["crossrefs"]
        )
        
        print(f"Headline: {test_case['headline']}")
        print(f"Source: {test_case['source']}")
        print(f"Cross-references: {', '.join(test_case['crossrefs'])}")
        print(f"\nResults:")
        print(f"  Fact-check score: {trust_result['fact_check_score']:.2f}")
        print(f"  Source reputation: {trust_result['source_reputation_score']:.2f}")
        print(f"  Multi-source score: {trust_result['multi_source_score']:.2f}")
        print(f"  Final trust score: {trust_result['trust_score']}")
        print(f"  Trust badge: {trust_result['trust_icon']} {trust_result['trust_badge']}")
        print(f"\nVerification details:")
        for key, value in trust_result['verification_details'].items():
            print(f"  {key.replace('_', ' ').title()}: {'✅' if value else '❌'}")

