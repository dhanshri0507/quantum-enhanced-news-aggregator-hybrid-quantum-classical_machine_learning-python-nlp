import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from typing import Tuple, Dict, Any
from preprocessing import TextPreprocessor, load_and_prepare_dataset

class ClassicalModelTrainer:
    def __init__(self):
        self.models = {}
        self.preprocessor = TextPreprocessor()
        self.is_trained = False
        
    def train_models(self, dataset_path: str = "News_Sentiment_Dataset.csv"):
        """Train all classical models on the dataset"""
        print("Loading and preparing dataset...")
        texts, sentiments = load_and_prepare_dataset(dataset_path)
        
        if not texts:
            raise ValueError("No data loaded from dataset")
        
        # Fit preprocessor
        self.preprocessor.fit_vectorizer(texts)
        
        # Prepare features
        X = self.preprocessor.prepare_features_batch(texts)
        y = np.array(sentiments)
        
        # Convert to binary classification (0: negative, 1: positive)
        y_binary = (y > 0.5).astype(int)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target distribution: {np.bincount(y_binary)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        # Initialize models
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(random_state=42, probability=True),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        # Train models
        results = {}
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} - Accuracy: {accuracy:.3f}, CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Print detailed results
        self._print_detailed_results(y_test, results)
        
        self.is_trained = True
        
        # Save models
        self.save_models()
        
        return results
    
    def _print_detailed_results(self, y_test: np.ndarray, results: Dict):
        """Print detailed classification results"""
        print("\n" + "="*50)
        print("DETAILED CLASSIFICATION RESULTS")
        print("="*50)
        
        for name, result in results.items():
            print(f"\n{name.upper()}:")
            print(f"Accuracy: {result['accuracy']:.3f}")
            print(f"Cross-validation: {result['cv_mean']:.3f} (+/- {result['cv_std'] * 2:.3f})")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_test, result['predictions'], 
                                    target_names=['Negative', 'Positive']))
    
    def predict_classical(self, text: str) -> Tuple[str, float]:
        """Make prediction using the best classical model"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Prepare features
        features = self.preprocessor.prepare_features(text)
        
        # Use Random Forest as the primary classical model (usually best for text)
        model = self.models['random_forest']
        
        # Make prediction
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0][1]  # Probability of positive class
        
        # Convert to label
        label = "Positive" if prediction == 1 else "Negative"
        
        return label, probability
    
    def predict_all_models(self, text: str) -> Dict[str, Tuple[str, float]]:
        """Make predictions using all classical models"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        features = self.preprocessor.prepare_features(text)
        predictions = {}
        
        for name, model in self.models.items():
            pred = model.predict([features])[0]
            proba = model.predict_proba([features])[0][1] if hasattr(model, 'predict_proba') else 0.5
            
            label = "Positive" if pred == 1 else "Negative"
            predictions[name] = (label, proba)
        
        return predictions
    
    def save_models(self, model_dir: str = "models"):
        """Save trained models to disk"""
        if not self.is_trained:
            raise ValueError("Models must be trained before saving")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            model_path = os.path.join(model_dir, f"{name}.joblib")
            joblib.dump(model, model_path)
            print(f"Saved {name} to {model_path}")
        
        # Save preprocessor
        preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"Saved preprocessor to {preprocessor_path}")
    
    def load_models(self, model_dir: str = "models"):
        """Load trained models from disk"""
        model_files = {
            'logistic_regression': 'logistic_regression.joblib',
            'svm': 'svm.joblib',
            'random_forest': 'random_forest.joblib'
        }
        
        self.models = {}
        for name, filename in model_files.items():
            model_path = os.path.join(model_dir, filename)
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
                print(f"Loaded {name} from {model_path}")
            else:
                print(f"Warning: {model_path} not found")
        
        # Load preprocessor
        preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
            self.is_trained = True
            print(f"Loaded preprocessor from {preprocessor_path}")
        else:
            print(f"Warning: {preprocessor_path} not found")

# Example usage
if __name__ == "__main__":
    trainer = ClassicalModelTrainer()
    
    # Train models
    results = trainer.train_models()
    
    # Test prediction
    test_text = "Government announces new economic stimulus package to boost recovery"
    label, confidence = trainer.predict_classical(test_text)
    print(f"\nTest prediction: '{test_text}' -> {label} (confidence: {confidence:.3f})")
    
    # Test all models
    all_predictions = trainer.predict_all_models(test_text)
    print("\nAll model predictions:")
    for model_name, (pred_label, pred_conf) in all_predictions.items():
        print(f"{model_name}: {pred_label} ({pred_conf:.3f})")
