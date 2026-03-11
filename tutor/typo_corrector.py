"""
Typo Correction Module for AI/ML/DL Terms
Provides fuzzy matching to automatically correct common typos in user queries
"""

import re
from difflib import get_close_matches
from typing import Tuple, Optional

# Comprehensive AI/ML/DL terminology dictionary
AI_ML_TERMINOLOGY = {
    # Core AI/ML Concepts
    "ai": "AI", "artificial intelligence": "Artificial Intelligence",
    "ml": "ML", "machine learning": "Machine Learning",
    "dl": "DL", "deep learning": "Deep Learning",
    
    # Neural Networks & Architectures
    "neural network": "Neural Network", "nn": "NN",
    "ann": "ANN", "artificial neural network": "Artificial Neural Network",
    "cnn": "CNN", "convolutional neural network": "Convolutional Neural Network",
    "rnn": "RNN", "recurrent neural network": "Recurrent Neural Network",
    "lstm": "LSTM", "long short term memory": "Long Short-Term Memory",
    "gru": "GRU", "gated recurrent unit": "Gated Recurrent Unit",
    "transformer": "Transformer",
    "attention": "Attention",
    "autoencoder": "Autoencoder",
    "gan": "GAN", "generative adversarial network": "Generative Adversarial Network",
    "bert": "BERT",
    "gpt": "GPT",
    "perceptron": "Perceptron",
    
    # Classification/Regression Algorithms
    "knn": "KNN", "k-nearest neighbors": "K-Nearest Neighbors",
    "k nearest": "K-Nearest",
    "svm": "SVM", "support vector machine": "Support Vector Machine",
    "decision tree": "Decision Tree",
    "random forest": "Random Forest",
    "naive bayes": "Naive Bayes",
    "linear regression": "Linear Regression",
    "logistic regression": "Logistic Regression",
    
    # Boosting/Ensemble Methods
    "boosting": "Boosting",
    "bagging": "Bagging",
    "adaboost": "AdaBoost",
    "xgboost": "XGBoost",
    "ensemble": "Ensemble",
    "gradient boosting": "Gradient Boosting",
    
    # Clustering
    "clustering": "Clustering",
    "k-means": "K-Means",
    "kmeans": "K-Means",
    "dbscan": "DBSCAN",
    
    # Dimensionality Reduction
    "pca": "PCA", "principal component analysis": "Principal Component Analysis",
    "dimensionality reduction": "Dimensionality Reduction",
    "t-sne": "t-SNE",
    
    # Core Concepts
    "supervised learning": "Supervised Learning",
    "unsupervised learning": "Unsupervised Learning",
    "reinforcement learning": "Reinforcement Learning",
    "semi-supervised": "Semi-Supervised",
    "transfer learning": "Transfer Learning",
    
    # Training Concepts
    "training": "Training",
    "testing": "Testing",
    "validation": "Validation",
    "overfitting": "Overfitting",
    "underfitting": "Underfitting",
    "cross validation": "Cross Validation",
    "epoch": "Epoch",
    "batch": "Batch",
    "gradient": "Gradient",
    "backpropagation": "Backpropagation",
    "loss": "Loss",
    
    # Optimization
    "optimizer": "Optimizer",
    "sgd": "SGD", "stochastic gradient descent": "Stochastic Gradient Descent",
    "adam": "Adam",
    "learning rate": "Learning Rate",
    "momentum": "Momentum",
    
    # Activation Functions
    "activation function": "Activation Function",
    "relu": "ReLU",
    "sigmoid": "Sigmoid",
    "tanh": "Tanh",
    "softmax": "Softmax",
    
    # Regularization
    "regularization": "Regularization",
    "dropout": "Dropout",
    "batch normalization": "Batch Normalization",
    "l1 regularization": "L1 Regularization",
    "l2 regularization": "L2 Regularization",
    "lasso": "Lasso",
    "ridge": "Ridge",
    
    # Network Components
    "layer": "Layer",
    "weight": "Weight",
    "bias": "Bias",
    "parameter": "Parameter",
    "hyperparameter": "Hyperparameter",
    "activation": "Activation",
    "pooling": "Pooling",
    "embedding": "Embedding",
    
    # Evaluation Metrics
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1 score": "F1 Score",
    "confusion matrix": "Confusion Matrix",
    "roc curve": "ROC Curve",
    "auc": "AUC",
    "mse": "MSE", "mean squared error": "Mean Squared Error",
    "rmse": "RMSE", "root mean squared error": "Root Mean Squared Error",
    "mae": "MAE", "mean absolute error": "Mean Absolute Error",
    
    # Data Related
    "dataset": "Dataset",
    "feature": "Feature",
    "label": "Label",
    "data preprocessing": "Data Preprocessing",
    "data augmentation": "Data Augmentation",
    "normalization": "Normalization",
    "standardization": "Standardization",
    
    # Other Models
    "model": "Model",
    "algorithm": "Algorithm",
    "function": "Function",
    "bias": "Bias",
    "variance": "Variance",
    "expert system": "Expert System",
    "fuzzy logic": "Fuzzy Logic",
    "genetic algorithm": "Genetic Algorithm",
}

# Common misspellings/typos for key terms
COMMON_TYPOS = {
    # Common 3-4 letter typos
    "ken": "knn",
    "knn ": "knn",
    "can": "cnn",
    "cnn ": "cnn",
    "rnn ": "rnn",
    "lstm ": "lstm",
    "gan ": "gan",
    "svm ": "svm",
    "pca ": "pca",
    "auc ": "auc",
    "mse ": "mse",
    
    # Common misspellings
    "neural net": "neural network",
    "convolutional": "convolutional",
    "recurrent": "recurrent",
    "backprob": "backpropagation",
    "back prop": "backpropagation",
    "relu ": "relu",
    "softmax ": "softmax",
    "tanh ": "tanh",
    "sigmod": "sigmoid",
    "sigmoid ": "sigmoid",
    "trainin": "training",
    "acuracy": "accuracy",
    "preccision": "precision",
    "overfitting ": "overfitting",
    "underfitting ": "underfitting",
    "hyperparameters": "hyperparameter",
    "optmizer": "optimizer",
    "learing rate": "learning rate",
    
    # Common abbreviation variations  
    "sgd ": "sgd",
    "adam ": "adam",
    "xgb": "xgboost",
    "rforest": "random forest",
    "r-forest": "random forest",
    "svm": "svm",
    "gbm": "gradient boosting",
    "dt": "decision tree",
    "lr": "logistic regression",
}


def correct_typos(text: str) -> Tuple[str, dict]:
    """
    Correct typos in the input text using fuzzy matching.
    
    Args:
        text: Input text to correct
        
    Returns:
        Tuple of (corrected_text, corrections_made) where corrections_made is a dict
        mapping original terms to corrected terms
    """
    corrected_text = text
    corrections_made = {}
    
    # First try exact common typo mappings
    for typo, correct in COMMON_TYPOS.items():
        pattern = r'\b' + re.escape(typo.rstrip()) + r'\b'
        matches = re.finditer(pattern, corrected_text, re.IGNORECASE)
        
        for match in matches:
            original = match.group()
            replacement = correct
            # Preserve case if original was capitalized
            if original[0].isupper():
                replacement = replacement.capitalize()
            corrections_made[original] = replacement
            corrected_text = corrected_text.replace(original, replacement)
    
    return corrected_text, corrections_made


def fuzzy_match_term(word: str, threshold: float = 0.75) -> Optional[str]:
    """
    Find the closest matching AI/ML term using fuzzy matching.
    
    Args:
        word: The word to match
        threshold: Minimum similarity score (0-1)
        
    Returns:
        The corrected term or None if no good match found
    """
    # Normalize input
    word_lower = word.lower().strip()
    
    # Check for exact match first
    if word_lower in AI_ML_TERMINOLOGY:
        return AI_ML_TERMINOLOGY[word_lower]
    
    # Try fuzzy matching
    close_matches = get_close_matches(
        word_lower, 
        AI_ML_TERMINOLOGY.keys(), 
        n=1, 
        cutoff=threshold
    )
    
    if close_matches:
        return AI_ML_TERMINOLOGY[close_matches[0]]
    
    return None


def correct_query(query: str, use_fuzzy: bool = True, fuzzy_threshold: float = 0.7) -> Tuple[str, dict]:
    """
    Correct typos and common misspellings in a user query.
    
    This function:
    1. Applies exact typo corrections
    2. Optionally applies fuzzy matching for additional corrections
    3. Returns the corrected query and a dict of corrections made
    
    Args:
        query: The user's input query
        use_fuzzy: Whether to apply fuzzy matching
        fuzzy_threshold: Threshold for fuzzy matching (0-1)
        
    Returns:
        Tuple of (corrected_query, corrections_dict)
    """
    corrected, exact_corrections = correct_typos(query)
    
    if use_fuzzy:
        # Split into words and handle fuzzy matching
        words = corrected.split()
        corrected_words = []
        
        for word in words:
            # Remove punctuation for matching but preserve it
            word_clean = re.sub(r'[^\w]', '', word)
            fuzzy_match = fuzzy_match_term(word_clean, threshold=fuzzy_threshold)
            
            if fuzzy_match and word_clean.lower() != fuzzy_match.lower():
                # Replace the word, preserving punctuation
                # Extract punctuation
                prefix = re.match(r'^[^\w]*', word).group()
                suffix = re.search(r'[^\w]*$', word).group()
                
                # Preserve case
                if word_clean and word_clean[0].isupper():
                    fuzzy_match = fuzzy_match.capitalize()
                
                corrected_words.append(prefix + fuzzy_match + suffix)
                exact_corrections[word_clean] = fuzzy_match
            else:
                corrected_words.append(word)
        
        corrected = ' '.join(corrected_words)
    
    return corrected, exact_corrections


def get_typo_correction_report(original: str, corrected: str, corrections: dict) -> str:
    """
    Generate a human-readable report of typo corrections made.
    
    Args:
        original: Original query
        corrected: Corrected query
        corrections: Dictionary of corrections applied
        
    Returns:
        A formatted report string
    """
    if not corrections:
        return ""
    
    report_lines = ["**Corrections applied:**"]
    for original_term, corrected_term in corrections.items():
        report_lines.append(f"  • '{original_term}' → '{corrected_term}'")
    
    return "\n".join(report_lines)
