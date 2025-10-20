import random
from datetime import datetime, timedelta

class TrustScoreGenerator:
    def __init__(self):
        # High reputation sources (score >= 0.7)
        self.high_reputation_sources = [
            "Reuters", "Associated Press", "BBC News", "The New York Times",
            "The Wall Street Journal", "The Washington Post", "NPR", "PBS News",
            "Bloomberg", "CNN", "ABC News", "CBS News", "NBC News"
        ]
        
        # Medium reputation sources (score 0.5-0.69)
        self.medium_reputation_sources = [
            "USA Today", "Time", "Newsweek", "Forbes", "Business Insider",
            "Los Angeles Times", "Chicago Tribune", "The Guardian"
        ]
        
        # Fact check categories with scores
        self.fact_check_categories = {
            "True": 1.0,
            "Mostly True": 0.8,
            "Half True": 0.5,
            "Mostly False": 0.2,
            "False": 0.0,
            "Unverified": 0.5
        }
        
        # News categories for diverse headlines
        self.news_categories = [
            "technology", "science", "politics", "health", "business",
            "environment", "education", "space", "economics"
        ]
        
        # Template headlines for each category
        self.headline_templates = {
            "technology": [
                "{} announces breakthrough in {} technology",
                "New {} innovation set to revolutionize {} industry",
                "{} researchers develop advanced {} system"
            ],
            "science": [
                "Scientists discover new evidence about {}",
                "Breakthrough study reveals {} phenomenon",
                "Research team makes significant {} finding"
            ],
            "politics": [
                "{} passes landmark legislation on {}",
                "Government announces new {} policy",
                "International agreement reached on {}"
            ],
            "health": [
                "Medical researchers develop new treatment for {}",
                "Study shows significant improvement in {} outcomes",
                "Healthcare breakthrough announced for {}"
            ],
            "business": [
                "{} reports record growth in {} sector",
                "Market analysis shows strong performance in {}",
                "Economic indicators point to {} expansion"
            ],
            "environment": [
                "New initiative launched to protect {} ecosystems",
                "Research confirms positive impact of {} conservation",
                "International coalition forms to address {}"
            ],
            "education": [
                "New study demonstrates effectiveness of {} approach",
                "Educational institutions adopt innovative {} methods",
                "Research shows improved outcomes with {} curriculum"
            ],
            "space": [
                "{} telescope makes groundbreaking discovery about {}",
                "Space agency announces mission to explore {}",
                "Astronomers confirm existence of {} phenomenon"
            ],
            "economics": [
                "Economic data shows strong growth in {} markets",
                "New report indicates positive trends in {} economy",
                "Analysis predicts sustained expansion in {} sector"
            ]
        }
        
        # Topic fillers for each category
        self.topic_fillers = {
            "technology": ["quantum computing", "AI", "renewable energy", "biotech", "nanotechnology"],
            "science": ["climate change", "marine biology", "particle physics", "genetics", "neuroscience"],
            "politics": ["climate policy", "healthcare reform", "education funding", "tax legislation", "foreign relations"],
            "health": ["cancer treatment", "mental health", "vaccine development", "chronic illness", "preventive care"],
            "business": ["technology", "manufacturing", "financial services", "healthcare", "consumer goods"],
            "environment": ["rainforest", "coral reef", "arctic", "wetland", "coastal"],
            "education": ["STEM", "early childhood", "digital literacy", "vocational training", "inclusive"],
            "space": ["distant galaxies", "exoplanets", "black holes", "solar system", "cosmic radiation"],
            "economics": ["emerging", "developed", "Asian", "European", "North American"]
        }

    def get_source_reputation(self, source):
        """Get reputation score for a source"""
        if source in self.high_reputation_sources:
            return 0.8 + random.random() * 0.2  # 0.8-1.0
        elif source in self.medium_reputation_sources:
            return 0.5 + random.random() * 0.2  # 0.5-0.7
        else:
            return random.random() * 0.5  # 0.0-0.5

    def generate_headline(self, category):
        """Generate a realistic headline for a given category"""
        templates = self.headline_templates.get(category, self.headline_templates["technology"])
        fillers = self.topic_fillers.get(category, self.topic_fillers["technology"])
        
        template = random.choice(templates)
        filler = random.choice(fillers)
        
        return template.format(random.choice(self.high_reputation_sources), filler)

    def generate_high_trust_news(self, count=10):
        """Generate news items with high trust scores"""
        news_items = []
        
        for _ in range(count):
            # Select a random category
            category = random.choice(self.news_categories)
            
            # Generate headline
            headline = self.generate_headline(category)
            
            # Select a high reputation source
            source = random.choice(self.high_reputation_sources)
            
            # Generate high fact check score (0.8-1.0)
            fc_score = 0.8 + random.random() * 0.2
            
            # Get source reputation (already high)
            rep_score = self.get_source_reputation(source)
            
            # Generate multi-source verification (high)
            multi_score = 0.7 + random.random() * 0.3
            
            # Calculate final trust score
            trust_score = round(100 * (0.5 * fc_score + 0.3 * rep_score + 0.2 * multi_score))
            
            # Determine trust level
            if trust_score >= 80:
                trust_level = "High"
            elif trust_score >= 50:
                trust_level = "Medium"
            else:
                trust_level = "Low"
            
            # Determine verification flags
            fact_checked = fc_score > 0.5
            reputable_source = rep_score >= 0.7
            multi_source_verified = multi_score > 0.0
            
            # Get fact check category
            if fc_score >= 0.8:
                fact_check_category = "True"
            elif fc_score >= 0.6:
                fact_check_category = "Mostly True"
            else:
                fact_check_category = "Unverified"
            
            # Generate a recent date
            date = (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d")
            
            news_items.append({
                "headline": headline,
                "source": source,
                "category": category,
                "date": date,
                "trust_score": trust_score,
                "trust_level": trust_level,
                "fact_check": {
                    "score": fc_score,
                    "category": fact_check_category
                },
                "source_reputation": rep_score,
                "multi_source_score": multi_score,
                "verification_flags": {
                    "fact_checked": fact_checked,
                    "reputable_source": reputable_source,
                    "multi_source_verified": multi_source_verified
                }
            })
        
        return news_items

    def print_news_item(self, news_item):
        """Print a news item in a formatted way"""
        print(f"Headline: {news_item['headline']}")
        print(f"Source: {news_item['source']}")
        print(f"Category: {news_item['category'].capitalize()}")
        print(f"Date: {news_item['date']}")
        print(f"Trust Score: {news_item['trust_score']}% ({news_item['trust_level']})")
        print(f"Fact Check: {news_item['fact_check']['category']} ({news_item['fact_check']['score']:.2f})")
        print(f"Source Reputation: {news_item['source_reputation']:.2f}")
        print(f"Multi-source Verification: {news_item['multi_source_score']:.2f}")
        print("Verification Flags:")
        print(f"  ✓ Fact Checked: {news_item['verification_flags']['fact_checked']}")
        print(f"  ✓ Reputable Source: {news_item['verification_flags']['reputable_source']}")
        print(f"  ✓ Multi-source Verified: {news_item['verification_flags']['multi_source_verified']}")
        print("-" * 80)

# Example usage
if __name__ == "__main__":
    generator = TrustScoreGenerator()
    
    print("Generating high-trust news headlines...")
    print("=" * 80)
    
    high_trust_news = generator.generate_high_trust_news(5)
    
    for i, news_item in enumerate(high_trust_news, 1):
        print(f"News Item #{i}:")
        generator.print_news_item(news_item)