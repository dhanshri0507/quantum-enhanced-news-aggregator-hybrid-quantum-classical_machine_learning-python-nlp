#!/usr/bin/env python3
"""
Training script for Quantum-Enhanced News Aggregator
This script trains both classical and quantum models on the news sentiment dataset
"""

import os
import sys
import time
from datetime import datetime
from typing import Tuple

from sklearn.metrics import confusion_matrix, classification_report
from preprocessing import load_and_prepare_dataset

def train_classical_models():
    """Train classical machine learning models"""
    print("="*60)
    print("TRAINING CLASSICAL MODELS")
    print("="*60)
    
    try:
        from classical_models import ClassicalModelTrainer
        
        trainer = ClassicalModelTrainer()
        results = trainer.train_models()
        
        print("\n✓ Classical models trained successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error training classical models: {e}")
        return False

def train_quantum_model():
    """Train quantum machine learning model"""
    print("\n" + "="*60)
    print("TRAINING QUANTUM MODEL")
    print("="*60)
    
    try:
        from quantum_model import QuantumModelTrainer
        
        trainer = QuantumModelTrainer()
        history = trainer.train()
        
        print("\n✓ Quantum model trained successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error training quantum model: {e}")
        return False

def test_models():
    """Test trained models with sample headlines"""
    print("\n" + "="*60)
    print("TESTING MODELS")
    print("="*60)
    
    test_headlines = [
        "Government announces new economic stimulus package to boost recovery",
        "Stock market crashes amid global economic uncertainty",
        "New technology breakthrough promises to revolutionize healthcare",
        "Climate change summit reaches historic agreement",
        "Local community celebrates successful fundraising campaign"
    ]
    
    try:
        from classical_models import ClassicalModelTrainer
        from quantum_model import QuantumModelTrainer
        from fact_check import TrustScoreCalculator
        
        # Load or train classical models
        classical_trainer = ClassicalModelTrainer()
        classical_trainer.load_models()
        
        # If classical models aren't trained, train them
        if not classical_trainer.is_trained:
            print("Classical models not found, training them first...")
            classical_trainer.train_models()
        
        # Load or train quantum model
        quantum_trainer = QuantumModelTrainer()
        quantum_trainer.quantum_classifier.load_model()
        
        # If quantum model isn't trained, train it
        if not quantum_trainer.quantum_classifier.is_trained:
            print("Quantum model not found, training it first...")
            quantum_trainer.train()
        
        trust_calculator = TrustScoreCalculator()
        
        print("Testing model predictions...")
        
        for i, headline in enumerate(test_headlines, 1):
            print(f"\n{i}. Headline: '{headline}'")
            
            # Classical prediction
            classical_label, classical_conf = classical_trainer.predict_classical(headline)
            print(f"   Classical: {classical_label} (confidence: {classical_conf:.3f})")
            
            # Quantum prediction
            quantum_label, quantum_conf = quantum_trainer.predict(headline)
            print(f"   Quantum: {quantum_label} (confidence: {quantum_conf:.3f})")
            
            # Trust score
            trust_result = trust_calculator.calculate_complete_trust_score(headline, "Reuters")
            print(f"   Trust: {trust_result['trust_icon']} {trust_result['trust_badge']} ({trust_result['trust_score']}%)")
            
            # Model agreement
            agreement = classical_label == quantum_label
            print(f"   Agreement: {'✓' if agreement else '✗'}")
        
        print("\n✓ Model testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing models: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints"""
    print("\n" + "="*60)
    print("TESTING API ENDPOINTS")
    print("="*60)
    
    try:
        import requests
        import json
        
        base_url = "http://localhost:8000"
        
        # Test health endpoint
        print("Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✓ Health endpoint working")
        else:
            print(f"✗ Health endpoint failed: {response.status_code}")
            return False
        
        # Test classification endpoint
        print("Testing classification endpoint...")
        test_data = {
            "headline": "Government announces new economic policy",
            "source": "Reuters"
        }
        
        response = requests.post(
            f"{base_url}/classify_and_verify",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Classification endpoint working")
            print(f"   Result: {result['sentiment_classical']} vs {result['sentiment_quantum']}")
            print(f"   Trust Score: {result['trust_score']}%")
        else:
            print(f"✗ Classification endpoint failed: {response.status_code}")
            return False
        
        print("\n✓ API testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing API: {e}")
        print("Make sure the API server is running (python main.py)")
        return False

def evaluate_confusion_matrices() -> bool:
    """Compute and display confusion matrices for classical and quantum models."""
    print("\n" + "="*60)
    print("CONFUSION MATRICES (CLASSICAL & QUANTUM)")
    print("="*60)

    try:
        from classical_models import ClassicalModelTrainer
        from quantum_model import QuantumModelTrainer

        # Load dataset and create binary labels
        texts, sentiments = load_and_prepare_dataset("News_Sentiment_Dataset.csv")
        if not texts:
            print("✗ No data loaded from dataset")
            return False
        import numpy as np
        y = np.array(sentiments)
        y_true = (y > 0.5).astype(int)

        # Load trainers and their persisted preprocessors/models
        classical_trainer = ClassicalModelTrainer()
        classical_trainer.load_models()
        if not classical_trainer.is_trained:
            print("Classical models not found; training before evaluation...")
            classical_trainer.train_models()

        quantum_trainer = QuantumModelTrainer()
        quantum_trainer.quantum_classifier.load_model()
        if not quantum_trainer.quantum_classifier.is_trained:
            print("Quantum model not found; training before evaluation...")
            quantum_trainer.train()

        # Prepare features using each model's own preprocessor
        X_classical = classical_trainer.preprocessor.prepare_features_batch(texts)
        X_quantum = quantum_trainer.quantum_classifier.preprocessor.prepare_features_batch(texts)

        # Classical predictions (Random Forest primary)
        rf_model = classical_trainer.models.get('random_forest')
        if rf_model is None:
            print("✗ Random Forest model not available")
            return False
        y_pred_classical = rf_model.predict(X_classical)

        # Quantum predictions (threshold 0.5)
        import numpy as np
        proba_quantum = quantum_trainer.quantum_classifier.model.predict(X_quantum, verbose=0).reshape(-1)
        y_pred_quantum = (proba_quantum > 0.5).astype(int)

        # Metrics: Confusion matrices and classification reports
        cm_classical = confusion_matrix(y_true, y_pred_classical)
        cm_quantum = confusion_matrix(y_true, y_pred_quantum)

        print("\n-- Classical (Random Forest) --")
        print("Confusion Matrix:")
        print(cm_classical)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_classical, target_names=['Negative', 'Positive']))

        print("\n-- Quantum Model --")
        print("Confusion Matrix:")
        print(cm_quantum)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_quantum, target_names=['Negative', 'Positive']))

        print("\n✓ Confusion matrices computed successfully!")
        return True

    except Exception as e:
        print(f"✗ Error computing confusion matrices: {e}")
        return False

def main():
    """Main training and testing pipeline"""
    print("🚀 QUANTUM-ENHANCED NEWS AGGREGATOR - TRAINING PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if dataset exists
    if not os.path.exists("News_Sentiment_Dataset.csv"):
        print("✗ Error: News_Sentiment_Dataset.csv not found!")
        print("Please ensure the dataset is in the current directory.")
        return False
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("✗ Error: .env file not found!")
        print("Please create .env file with your API keys.")
        return False
    
    success_count = 0
    total_tests = 4
    
    # Train classical models
    if train_classical_models():
        success_count += 1
    
    # Train quantum model
    if train_quantum_model():
        success_count += 1
    
    # Test models
    if test_models():
        success_count += 1
    
    # Confusion matrices for both models
    if evaluate_confusion_matrices():
        success_count += 1

    # Test API (optional - requires server to be running)
    print("\n" + "="*60)
    print("API TESTING (Optional)")
    print("="*60)
    print("To test API endpoints, start the server with: python main.py")
    print("Then run this script again or test manually at http://localhost:8000")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Completed: {success_count}/{total_tests} tests")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_count >= 3:  # At least models trained and tested
        print("\n🎉 Training completed successfully!")
        print("\nNext steps:")
        print("1. Start the API server: python main.py")
        print("2. Open web UI: http://localhost:8000/ui")
        print("3. Test with real news headlines")
        return True
    else:
        print("\n⚠️ Some issues encountered. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)