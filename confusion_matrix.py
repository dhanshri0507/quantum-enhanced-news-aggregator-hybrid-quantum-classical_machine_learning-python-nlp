#!/usr/bin/env python3
"""
Generate confusion matrices with improved spacing between predicted label and accuracy text
"""

import os
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.metrics import confusion_matrix, classification_report # type: ignore
from preprocessing import load_and_prepare_dataset

def create_confusion_matrix_plot(y_true, y_pred, model_name, save_path):
    """Create and save a confusion matrix plot with better spacing"""
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(cm) / np.sum(cm)
    
    # Create figure with larger size and more vertical space
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create heatmap with better styling
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'},
                ax=ax,
                annot_kws={'size': 16, 'weight': 'bold'})
    
    # Customize plot with better spacing
    ax.set_title(f'Confusion Matrix - {model_name}', 
                fontsize=20, fontweight='bold', pad=40)
    ax.set_xlabel('Predicted Label', fontsize=16, fontweight='bold', labelpad=20)
    ax.set_ylabel('True Label', fontsize=16, fontweight='bold', labelpad=20)
    
    # Add accuracy text with much more distance from the plot
    fig.text(0.5, 0.02, f'Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)', 
            ha='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Adjust layout with more padding and space
    plt.tight_layout(pad=4.0)
    
    # Add extra space at the bottom by adjusting subplot parameters
    plt.subplots_adjust(bottom=0.15)
    
    # Save with high quality
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.8)
    plt.close()
    
    print(f"✓ Confusion matrix saved: {save_path}")
    return cm, accuracy

def generate_confusion_matrices():
    """Generate and save confusion matrices for both models"""
    
    print("🚀 GENERATING CONFUSION MATRICES (IMPROVED SPACING)")
    print("="*60)
    
    try:
        # Load dataset and create binary labels
        print("📊 Loading dataset...")
        texts, sentiments = load_and_prepare_dataset("News_Sentiment_Dataset.csv")
        if not texts:
            print("✗ No data loaded from dataset")
            return False
        
        y = np.array(sentiments)
        y_true = (y > 0.5).astype(int)
        print(f"✓ Dataset loaded: {len(texts)} samples")
        
        # Load classical models
        print("\n🔧 Loading classical models...")
        from classical_models import ClassicalModelTrainer
        classical_trainer = ClassicalModelTrainer()
        classical_trainer.load_models()
        
        if not classical_trainer.is_trained:
            print("Classical models not found; training before evaluation...")
            classical_trainer.train_models()
        
        # Load quantum model
        print("🔬 Loading quantum model...")
        from quantum_model import QuantumModelTrainer
        quantum_trainer = QuantumModelTrainer()
        quantum_trainer.quantum_classifier.load_model()
        
        if not quantum_trainer.quantum_classifier.is_trained:
            print("Quantum model not found; training before evaluation...")
            quantum_trainer.train()
        
        # Prepare features using each model's own preprocessor
        print("\n⚙️ Preparing features...")
        X_classical = classical_trainer.preprocessor.prepare_features_batch(texts)
        X_quantum = quantum_trainer.quantum_classifier.preprocessor.prepare_features_batch(texts)
        
        # Classical predictions (Random Forest primary)
        print("📈 Generating classical predictions...")
        rf_model = classical_trainer.models.get('random_forest')
        if rf_model is None:
            print("✗ Random Forest model not available")
            return False
        
        y_pred_classical = rf_model.predict(X_classical)
        
        # Quantum predictions (threshold 0.5)
        print("🔮 Generating quantum predictions...")
        proba_quantum = quantum_trainer.quantum_classifier.model.predict(X_quantum, verbose=0).reshape(-1)
        y_pred_quantum = (proba_quantum > 0.5).astype(int)
        
        # Create output directory if it doesn't exist
        output_dir = "confusion_matrices"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✓ Created directory: {output_dir}")
        
        # Generate and save confusion matrices
        print("\n📊 Generating confusion matrices...")
        
        # Classical model confusion matrix
        cm_classical, acc_classical = create_confusion_matrix_plot(
            y_true, y_pred_classical, 
            "Classical Model (Random Forest)",
            os.path.join(output_dir, "classical_confusion_matrix_spaced.png")
        )
        
        # Quantum model confusion matrix
        cm_quantum, acc_quantum = create_confusion_matrix_plot(
            y_true, y_pred_quantum,
            "Quantum Model (VQC)",
            os.path.join(output_dir, "quantum_confusion_matrix_spaced.png")
        )
        
        # Print detailed results
        print("\n" + "="*60)
        print("CONFUSION MATRIX RESULTS")
        print("="*60)
        
        print(f"\n🔧 Classical Model (Random Forest):")
        print(f"   Accuracy: {acc_classical:.3f}")
        print(f"   Confusion Matrix:")
        print(f"   {cm_classical}")
        
        print(f"\n🔮 Quantum Model (VQC):")
        print(f"   Accuracy: {acc_quantum:.3f}")
        print(f"   Confusion Matrix:")
        print(f"   {cm_quantum}")
        
        # Performance comparison
        print(f"\n📊 Performance Comparison:")
        print(f"   Classical Model Accuracy: {acc_classical:.3f}")
        print(f"   Quantum Model Accuracy:   {acc_quantum:.3f}")
        print(f"   Difference:               {abs(acc_classical - acc_quantum):.3f}")
        
        if acc_quantum > acc_classical:
            print(f"   🏆 Quantum model performs better by {acc_quantum - acc_classical:.3f}")
        elif acc_classical > acc_quantum:
            print(f"   🏆 Classical model performs better by {acc_classical - acc_quantum:.3f}")
        else:
            print(f"   🤝 Both models perform equally well")
        
        print(f"\n✅ Improved confusion matrices saved successfully!")
        print(f"   📁 Directory: {output_dir}/")
        print(f"   📄 Files:")
        print(f"      - classical_confusion_matrix_spaced.png")
        print(f"      - quantum_confusion_matrix_spaced.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating confusion matrices: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = generate_confusion_matrices()
    if success:
        print("\n🎉 Improved confusion matrix generation completed successfully!")
    else:
        print("\n💥 Confusion matrix generation failed!")