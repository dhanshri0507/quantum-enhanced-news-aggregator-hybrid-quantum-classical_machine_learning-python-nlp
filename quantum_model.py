import numpy as np
import pandas as pd
import pennylane as qml
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, List
import joblib
import os
from preprocessing import TextPreprocessor, load_and_prepare_dataset
from config import QUANTUM_QUBITS, PCA_COMPONENTS

class QuantumLayer(keras.layers.Layer):
    """Enhanced quantum layer with improved expressivity and architecture"""
    
    def __init__(self, n_qubits, n_layers, **kwargs):
        super(QuantumLayer, self).__init__(**kwargs)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = qml.device('default.qubit', wires=n_qubits)
        
        # Define quantum circuit
        self.qnode = qml.QNode(self._quantum_circuit, self.device, interface='tf')
        
    def _quantum_circuit(self, inputs, weights):
        """Enhanced quantum circuit with better expressivity"""
        # Improved data encoding with multiple rotations
        for i in range(self.n_qubits):
            qml.RY(inputs[i] * np.pi, wires=i)
            qml.RZ(inputs[i] * np.pi * 0.5, wires=i)
        
        # Multi-layer variational ansatz with more parameters
        for layer in range(self.n_layers):
            # Rotation layer with more gates
            for i in range(self.n_qubits):
                qml.RX(weights[layer*4, i], wires=i)
                qml.RY(weights[layer*4+1, i], wires=i)
                qml.RZ(weights[layer*4+2, i], wires=i)
                qml.RX(weights[layer*4+3, i], wires=i)
            
            # Enhanced entangling layer with more connectivity
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Additional entangling gates for better connectivity
            if self.n_qubits > 2:
                qml.CNOT(wires=[0, self.n_qubits-1])
            if self.n_qubits > 3:
                qml.CNOT(wires=[1, self.n_qubits-2])
        
        # Multi-qubit readout for better information extraction
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def build(self, input_shape):
        # Create weights for enhanced circuit (4 parameters per layer)
        self.quantum_weights = self.add_weight(
            name='quantum_weights',
            shape=(self.n_layers * 4, self.n_qubits),
            initializer='glorot_uniform',  # Better initialization
            trainable=True
        )
        super(QuantumLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Ensure inputs are the right shape and dtype
        inputs = tf.reshape(inputs, [-1, self.n_qubits])
        inputs = tf.cast(inputs, tf.float32)
        
        # Apply quantum circuit to each input
        batch_size = tf.shape(inputs)[0]
        
        # Use tf.map_fn to apply quantum circuit to each sample
        def apply_quantum(x):
            result = self.qnode(x, self.quantum_weights)
            # Convert from float64 to float32 and handle multi-qubit output
            return tf.cast(tf.stack(result), tf.float32)
        
        outputs = tf.map_fn(apply_quantum, inputs, fn_output_signature=tf.TensorSpec(shape=(self.n_qubits,), dtype=tf.float32))
        return outputs
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_qubits)

class VariationalQuantumClassifier:
    def __init__(self, n_qubits: int = QUANTUM_QUBITS, n_layers: int = 4):  # Increased layers
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.is_trained = False
    
    def _create_model(self, input_dim: int):
        """Create the enhanced hybrid quantum-classical model"""
        # Input layer
        inputs = keras.Input(shape=(input_dim,), name='input')
        
        # Enhanced classical preprocessing layers
        x = keras.layers.Dense(32, activation='relu')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        
        x = keras.layers.Dense(24, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        
        x = keras.layers.Dense(self.n_qubits, activation='tanh')(x)
        
        # Custom quantum layer
        quantum_layer = QuantumLayer(self.n_qubits, self.n_layers, name='quantum_layer')
        quantum_output = quantum_layer(x)

        # Enhanced classical post-processing
        x = keras.layers.Dense(16, activation='relu')(quantum_output)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        
        x = keras.layers.Dense(8, activation='relu')(x)
        x = keras.layers.Dropout(0.1)(x)
        
        outputs = keras.layers.Dense(1, activation='sigmoid', name='output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def train_quantum_model(self, dataset_path: str = "News_Sentiment_Dataset.csv", 
                        epochs: int = 50, batch_size: int = 16):  # Increased epochs, smaller batch
        """Train the enhanced quantum model"""
        print("Loading and preparing dataset for quantum training...")
        texts, sentiments = load_and_prepare_dataset(dataset_path)
        
        if not texts:
            raise ValueError("No data loaded from dataset")
        
        # Fit preprocessor
        self.preprocessor.fit_vectorizer(texts)
        
        # Prepare features
        X = self.preprocessor.prepare_features_batch(texts)
        y = np.array(sentiments)
        
        # Convert to binary classification
        y_binary = (y > 0.5).astype(float)
        
        print(f"Quantum training data shape: {X.shape}")
        print(f"Target distribution: {np.bincount(y_binary.astype(int))}")
        
        # Create model
        self.model = self._create_model(X.shape[1])
        
        # Enhanced optimizer with learning rate scheduling
        initial_learning_rate = 0.01
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100,
            decay_rate=0.96,
            staircase=True
        )
        
        # Compile model with enhanced settings
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("Enhanced quantum model architecture:")
        self.model.summary()
        
        # Early stopping and callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        print(f"\nTraining enhanced quantum model for {epochs} epochs...")
        history = self.model.fit(
            X, y_binary,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        return history
    
    def predict_quantum(self, text: str) -> Tuple[str, float]:
        """Make prediction using the quantum model"""
        if not self.is_trained:
            raise ValueError("Quantum model must be trained before making predictions")
        
        # Prepare features
        features = self.preprocessor.prepare_features(text)
        
        # Make prediction
        prediction_proba = self.model.predict([features.reshape(1, -1)], verbose=0)[0][0]
        
        # Convert to label and probability
        label = "Positive" if prediction_proba > 0.5 else "Negative"
        probability = prediction_proba if prediction_proba > 0.5 else 1 - prediction_proba
        
        return label, probability
    
    def predict_batch_quantum(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Make predictions for a batch of texts"""
        if not self.is_trained:
            raise ValueError("Quantum model must be trained before making predictions")
        
        # Prepare features
        features = self.preprocessor.prepare_features_batch(texts)
        
        # Make predictions
        predictions_proba = self.model.predict(features, verbose=0)
        
        results = []
        for proba in predictions_proba:
            label = "Positive" if proba[0] > 0.5 else "Negative"
            probability = proba[0] if proba[0] > 0.5 else 1 - proba[0]
            results.append((label, probability))
        
        return results
    
    def save_model(self, model_dir: str = "models"):
        """Save the trained quantum model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save TensorFlow model
        model_path = os.path.join(model_dir, "quantum_model")
        self.model.save(model_path)
        print(f"Saved quantum model to {model_path}")
        
        # Save preprocessor
        preprocessor_path = os.path.join(model_dir, "quantum_preprocessor.joblib")
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"Saved quantum preprocessor to {preprocessor_path}")
    
    def load_model(self, model_dir: str = "models"):
        """Load the trained quantum model"""
        model_path = os.path.join(model_dir, "quantum_model")
        preprocessor_path = os.path.join(model_dir, "quantum_preprocessor.joblib")
        
        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            # Load TensorFlow model
            self.model = keras.models.load_model(model_path)
            
            # Load preprocessor
            self.preprocessor = joblib.load(preprocessor_path)
            
            self.is_trained = True
            print(f"Loaded quantum model from {model_path}")
            print(f"Loaded quantum preprocessor from {preprocessor_path}")
        else:
            print(f"Warning: Model files not found in {model_dir}")

class QuantumModelTrainer:
    """Simplified quantum model trainer for demonstration"""
    
    def __init__(self):
        self.quantum_classifier = VariationalQuantumClassifier()
    
    def train(self, dataset_path: str = "News_Sentiment_Dataset.csv"):
        """Train the quantum model"""
        return self.quantum_classifier.train_quantum_model(dataset_path)
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Make prediction using quantum model"""
        return self.quantum_classifier.predict_quantum(text)

# Example usage
if __name__ == "__main__":
    # Initialize quantum trainer
    quantum_trainer = QuantumModelTrainer()
    
    # Train quantum model (use smaller dataset for demo)
    print("Training quantum model...")
    history = quantum_trainer.train()
    
    # Test prediction
    test_text = "Government announces new economic stimulus package to boost recovery"
    label, confidence = quantum_trainer.predict(test_text)
    print(f"\nQuantum prediction: '{test_text}' -> {label} (confidence: {confidence:.3f})")
    
    # Compare with classical (if available)
    try:
        from classical_models import ClassicalModelTrainer
        classical_trainer = ClassicalModelTrainer()
        classical_trainer.load_models()
        
        classical_label, classical_confidence = classical_trainer.predict_classical(test_text)
        print(f"Classical prediction: {classical_label} (confidence: {classical_confidence:.3f})")
        
        print(f"\nModel agreement: {'✓' if label == classical_label else '✗'}")
        
    except Exception as e:
        print(f"Could not compare with classical model: {e}")
