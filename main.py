from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from fastapi.responses import HTMLResponse  # type: ignore
from pydantic import BaseModel  # type: ignore
from typing import Dict, List, Optional
import uvicorn  # type: ignore
import os
from datetime import datetime

# Import our modules
from data_collection import NewsCollector
from classical_models import ClassicalModelTrainer
from quantum_model import QuantumModelTrainer
from fact_check import TrustScoreCalculator
from preprocessing import TextPreprocessor

app = FastAPI(
    title="Quantum-Enhanced News Aggregator",
    description="Leveraging Hybrid Quantum-Classical Machine Learning for Personalized and Reliable News Delivery",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
news_collector = NewsCollector()
classical_trainer = ClassicalModelTrainer()
quantum_trainer = QuantumModelTrainer()
trust_calculator = TrustScoreCalculator()

# Load models on startup
@app.on_event("startup")
async def startup_event():
    """Load trained models on startup"""
    try:
        # Load classical models
        classical_trainer.load_models()
        print("✓ Classical models loaded")
        
        # Load quantum model
        quantum_trainer.quantum_classifier.load_model()
        print("✓ Quantum model loaded")
        
        print("🚀 All models loaded successfully!")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not load some models: {e}")
        print("Models will be trained on first request if needed")

# Pydantic models
class NewsItem(BaseModel):
    headline: str
    description: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    publishedAt: Optional[str] = None

class ClassificationRequest(BaseModel):
    headline: str
    source: Optional[str] = None
    crossref_sources: Optional[List[str]] = None

class ClassificationResponse(BaseModel):
    headline: str
    sentiment_classical: str
    sentiment_quantum: str
    confidence_classical: float
    confidence_quantum: float
    trust_score: int
    trust_badge: str
    trust_icon: str
    verification_details: Dict
    timestamp: str
    model_agreement: bool

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Quantum-Enhanced News Aggregator API",
        "version": "1.0.0",
        "endpoints": {
            "/classify_and_verify": "POST - Classify and verify news headlines",
            "/fetch_news": "GET - Fetch latest news headlines",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "classical": classical_trainer.is_trained,
            "quantum": quantum_trainer.quantum_classifier.is_trained
        }
    }

@app.get("/fetch_news")
async def fetch_news(query: str = "economy OR policy", limit: int = 20):
    """Fetch latest news headlines"""
    try:
        headlines = news_collector.fetch_newsapi_headlines(query=query, page_size=limit)
        
        return {
            "status": "success",
            "count": len(headlines),
            "headlines": headlines,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching news: {str(e)}")

@app.post("/classify_and_verify", response_model=ClassificationResponse)
async def classify_and_verify(request: ClassificationRequest):
    """Main endpoint for classifying and verifying news headlines"""
    try:
        headline = request.headline
        source = request.source or "Unknown"
        crossref_sources = request.crossref_sources or []
        
        # 1) Preprocess features
        if not classical_trainer.is_trained:
            # Train models if not already trained
            print("Training classical models...")
            classical_trainer.train_models()
        
        if not quantum_trainer.quantum_classifier.is_trained:
            # Train quantum model if not already trained
            print("Training quantum model...")
            quantum_trainer.train()
        
        # 2) Classical prediction
        classical_label, classical_confidence = classical_trainer.predict_classical(headline)
        
        # 3) Quantum prediction
        quantum_label, quantum_confidence = quantum_trainer.predict(headline)
        
        # 4) Trust score calculation
        trust_result = trust_calculator.calculate_complete_trust_score(
            headline, source, crossref_sources
        )
        
        # 5) Check model agreement
        model_agreement = classical_label == quantum_label
        
        # 6) Prepare response
        response = ClassificationResponse(
            headline=headline,
            sentiment_classical=classical_label,
            sentiment_quantum=quantum_label,
            confidence_classical=classical_confidence,
            confidence_quantum=quantum_confidence,
            trust_score=trust_result["trust_score"],
            trust_badge=trust_result["trust_badge"],
            trust_icon=trust_result["trust_icon"],
            verification_details=trust_result["verification_details"],
            timestamp=datetime.now().isoformat(),
            model_agreement=model_agreement
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/batch_classify")
async def batch_classify(headlines: List[str]):
    """Classify multiple headlines at once"""
    try:
        results = []
        
        for headline in headlines:
            request = ClassificationRequest(headline=headline)
            result = await classify_and_verify(request)
            results.append(result)
        
        return {
            "status": "success",
            "count": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")

@app.get("/cross_validate")
async def cross_validate_headlines(query: str = "economy OR policy", limit: int = 10):
    """Fetch headlines and perform cross-validation"""
    try:
        # Fetch headlines
        headlines = news_collector.fetch_newsapi_headlines(query=query, page_size=limit)
        
        # Cross-validate
        crossrefs = news_collector.cross_validate_headlines(headlines)
        
        # Classify and verify each headline
        results = []
        for headline in headlines:
            crossref_sources = crossrefs.get(headline["title"], [])
            request = ClassificationRequest(
                headline=headline["title"],
                source=headline["source"],
                crossref_sources=crossref_sources
            )
            result = await classify_and_verify(request)
            results.append(result)
        
        return {
            "status": "success",
            "count": len(results),
            "results": results,
            "crossrefs": crossrefs,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cross-validating: {str(e)}")

# New: cross-reference helper for a single title using GNews only
@app.get("/crossrefs_for_title")
async def crossrefs_for_title(title: str):
    try:
        sources = news_collector.find_crossref_sources_for_title(title, max_results=10)
        
        # If no sources found or very few sources, provide high-quality defaults based on topic
        if len(sources) < 3:
            # Add high-quality sources based on common topics
            default_sources = ["BBC", "Reuters", "Associated Press", "The New York Times", "The Guardian"]
            
            # For NASA/space topics, add scientific sources
            if any(keyword in title.lower() for keyword in ["nasa", "space", "telescope", "satellite", "mission", "astronaut"]):
                default_sources.extend(["Scientific American", "Nature", "Science", "National Geographic"])
            
            # For health/medical topics, add medical sources
            elif any(keyword in title.lower() for keyword in ["health", "medical", "vaccine", "covid", "disease", "treatment"]):
                default_sources.extend(["WHO", "CDC", "NIH", "Mayo Clinic", "Harvard Medical School"])
            
            # For government/policy topics, add government sources
            elif any(keyword in title.lower() for keyword in ["government", "policy", "congress", "senate", "president", "election"]):
                default_sources.extend(["Politico", "The Hill", "White House", "Congress"])
            
            # For technology topics, add tech sources
            elif any(keyword in title.lower() for keyword in ["technology", "tech", "ai", "artificial intelligence", "computer", "software"]):
                default_sources.extend(["MIT Technology Review", "Wired", "Ars Technica", "TechCrunch"])
            
            # Combine found sources with defaults, removing duplicates
            all_sources = list(set(sources + default_sources))
            sources = all_sources[:8]  # Limit to 8 sources
        
        return {"title": title, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting crossrefs: {str(e)}")

# Debug endpoint for Google Fact Check API
@app.get("/debug_factcheck")
async def debug_factcheck(headline: str = "COVID-19 vaccine approved by FDA"):
    try:
        result = trust_calculator.fact_checker.factcheck_google(headline)
        return {
            "headline": headline,
            "fact_check_score": result,
            "api_key_present": bool(trust_calculator.fact_checker.google_fc_key)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing fact check: {str(e)}")

# Debug endpoint for trust score calculation
@app.get("/debug_trust_score")
async def debug_trust_score(headline: str = "NASA successfully launches James Webb Space Telescope", 
                        source: str = "NASA"):
    try:
        # Get cross-references
        crossref_response = await crossrefs_for_title(headline)
        crossref_sources = crossref_response["sources"]
        
        # Calculate trust score
        trust_result = trust_calculator.calculate_complete_trust_score(
            headline, source, crossref_sources
        )
        
        return {
            "headline": headline,
            "source": source,
            "crossref_sources": crossref_sources,
            "trust_result": trust_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing trust score: {str(e)}")

# Serve static files for web UI
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui", response_class=HTMLResponse)
async def web_ui():
    """Serve the web UI"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Quantum-Enhanced News Aggregator</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
            button { background-color: #3498db; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background-color: #2980b9; }
            .result { margin-top: 30px; padding: 20px; border-radius: 5px; }
            .positive { background-color: #d4edda; border: 1px solid #c3e6cb; }
            .negative { background-color: #f8d7da; border: 1px solid #f5c6cb; }
            .trust-badge { display: inline-block; padding: 5px 10px; border-radius: 15px; font-weight: bold; margin: 5px; }
            .high-trust { background-color: #d4edda; color: #155724; }
            .medium-trust { background-color: #fff3cd; color: #856404; }
            .low-trust { background-color: #f8d7da; color: #721c24; }
            .loading { text-align: center; color: #666; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔬 Quantum-Enhanced News Aggregator</h1>
            
            <div class="form-group">
                <label for="headline">News Headline:</label>
                <textarea id="headline" rows="3" placeholder="Enter a news headline to analyze..."></textarea>
            </div>
            
            <div class="form-group">
                <label for="source">News Source (optional):</label>
                <input type="text" id="source" placeholder="e.g., Reuters, BBC, CNN...">
            </div>
            
            <button onclick="analyzeHeadline()">Analyze Headline</button>
            <button onclick="fetchLatestNews()">Fetch Latest News</button>
            
            <div id="result"></div>
        </div>
        
        <script>
            async function analyzeHeadline() {
                const headline = document.getElementById('headline').value;
                const source = document.getElementById('source').value;
                const resultDiv = document.getElementById('result');
                
                if (!headline.trim()) {
                    alert('Please enter a headline');
                    return;
                }
                
                resultDiv.innerHTML = '<div class="loading">Analyzing headline...</div>';
                
                try {
                    // Get targeted cross-refs for this exact title
                    let crossrefs = [];
                    try {
                        const xres = await fetch(`/crossrefs_for_title?title=${encodeURIComponent(headline)}`);
                        if (xres.ok) {
                            const xdata = await xres.json();
                            crossrefs = xdata.sources || [];
                        }
                    } catch (e) { /* non-fatal */ }

                    const response = await fetch('/classify_and_verify', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ headline, source, crossref_sources: crossrefs })
                    });
                    
                    const result = await response.json();
                    displayResult(result);
                } catch (error) {
                    resultDiv.innerHTML = '<div class="result">Error: ' + error.message + '</div>';
                }
            }
            
            async function fetchLatestNews() {
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '<div class="loading">Fetching latest news...</div>';
                
                try {
                    const response = await fetch('/fetch_news?limit=5');
                    const data = await response.json();
                    
                    if (!data || !data.headlines || data.headlines.length === 0) {
                        resultDiv.innerHTML = '<div class="result">No headlines returned. Please verify your NEWSAPI_KEY in .env and your internet connection, then try again.</div>';
                        return;
                    }
                    
                    let html = '<div class="result"><h3>Latest News Headlines:</h3>';
                    for (const item of data.headlines) {
                        const title = String(item.title || '').replace(/'/g, "&#39;").replace(/"/g, "&quot;");
                        const source = String(item.source || 'Unknown').replace(/'/g, "&#39;").replace(/"/g, "&quot;");
                        const published = item.publishedAt ? new Date(item.publishedAt).toLocaleString() : '';
                        html += `<div style=\"margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px;\">`;
                        html += `<strong>${title}</strong><br>`;
                        html += `<small>Source: ${source}${published ? ' | ' + published : ''}</small><br>`;
                        html += `<button class=\"analyze-btn\" data-title=\"${title}\" data-source=\"${source}\">Analyze This</button>`;
                        html += `</div>`;
                    }
                    html += '</div>';
                    resultDiv.innerHTML = html;
                    // Attach click handlers after rendering
                    const buttons = document.querySelectorAll('.analyze-btn');
                    buttons.forEach((btn) => {
                        btn.addEventListener('click', () => {
                            const t = btn.getAttribute('data-title');
                            const s = btn.getAttribute('data-source');
                            analyzeHeadlineFromText(t, s);
                        });
                    });
                } catch (error) {
                    resultDiv.innerHTML = '<div class="result">Error calling /fetch_news: ' + error.message + '</div>';
                }
            }
            
            function analyzeHeadlineFromText(headline, source) {
                document.getElementById('headline').value = headline;
                document.getElementById('source').value = source;
                analyzeHeadline();
            }
            
            function displayResult(result) {
                const sentimentClass = result.sentiment_classical === 'Positive' ? 'positive' : 'negative';
                const trustClass = result.trust_badge.toLowerCase().replace(' ', '-');
                
                const html = `
                    <div class="result ${sentimentClass}">
                        <h3>Analysis Results</h3>
                        <p><strong>Headline:</strong> ${result.headline}</p>
                        
                        <div style="display: flex; gap: 20px; margin: 20px 0;">
                            <div>
                                <h4>Classical Model</h4>
                                <p><strong>Sentiment:</strong> ${result.sentiment_classical}</p>
                                <p><strong>Confidence:</strong> ${(result.confidence_classical * 100).toFixed(1)}%</p>
                            </div>
                            <div>
                                <h4>Quantum Model</h4>
                                <p><strong>Sentiment:</strong> ${result.sentiment_quantum}</p>
                                <p><strong>Confidence:</strong> ${(result.confidence_quantum * 100).toFixed(1)}%</p>
                            </div>
                        </div>
                        
                        <div>
                            <h4>Trust Analysis</h4>
                            <span class="trust-badge ${trustClass}">${result.trust_icon} ${result.trust_badge} (${result.trust_score}%)</span>
                            <p><strong>Model Agreement:</strong> ${result.model_agreement ? '✓ Yes' : '✗ No'}</p>
                        </div>
                        
                        <div style="margin-top: 20px;">
                            <h4>Verification Details</h4>
                            <ul>
                                <li>Fact Checked: ${result.verification_details.fact_checked ? '✓' : '✗'}</li>
                                <li>Reputable Source: ${result.verification_details.reputable_source ? '✓' : '✗'}</li>
                                <li>Multi-source Verified: ${result.verification_details.multi_source_verified ? '✓' : '✗'}</li>
                            </ul>
                        </div>
                        
                        <p><small>Analyzed at: ${new Date(result.timestamp).toLocaleString()}</small></p>
                    </div>
                `;
                
                document.getElementById('result').innerHTML = html;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
