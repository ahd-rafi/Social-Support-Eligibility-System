"""Eligibility classifier using scikit-learn."""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.config import settings

logger = logging.getLogger(__name__)


class EligibilityClassifier:
    """Eligibility classifier using HistGradientBoostingClassifier.
    
    Predicts eligibility category based on applicant features:
    - approved_financial: Qualifies for financial support
    - approved_enablement: Qualifies for enablement programs only
    - soft_decline: Does not qualify
    """
    
    # Feature names in order
    FEATURE_NAMES = [
        "monthly_income",
        "employment_months",
        "household_size",
        "num_dependents",
        "income_per_family",
        "total_assets",
        "total_liabilities",
        "net_worth",
        "debt_to_income",
        "credit_score",
        "missed_payments",
        "age",
    ]
    
    # Category mapping
    CATEGORIES = ["approved_financial", "approved_enablement", "soft_decline"]
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize classifier.
        
        Args:
            model_path: Path to saved model (optional)
        """
        self.model = None
        self.model_path = model_path or (settings.model_dir / "eligibility_classifier.pkl")
        self.is_trained = False

    def create_model(self) -> HistGradientBoostingClassifier:
        """Create a new HistGradientBoostingClassifier.
        
        Returns:
            Configured classifier
        """
        model = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=5,
            random_state=42,
            verbose=0,
        )
        
        logger.info("Created HistGradientBoostingClassifier")
        return model

    def prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare feature vector for prediction.
        
        Args:
            features: Dict with feature names and values
            
        Returns:
            Feature vector as numpy array
        """
        # Extract features in correct order
        feature_vector = []
        for feature_name in self.FEATURE_NAMES:
            value = features.get(feature_name)
            
            # Handle missing values
            if value is None or pd.isna(value):
                value = 0  # Default to 0 for missing values
            
            # Handle employment_status (categorical) - convert to months
            if feature_name == "employment_months" and isinstance(value, str):
                value = 0 if value == "unemployed" else 12  # Default to 12 months if employed
            
            feature_vector.append(float(value))
        
        return np.array(feature_vector).reshape(1, -1)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
    ) -> Dict[str, Any]:
        """Train the classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            cv: Number of cross-validation folds
            
        Returns:
            Training metrics
        """
        logger.info(f"Training classifier on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Create model
        self.model = self.create_model()
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='accuracy')
        
        # Test set performance
        y_pred = self.model.predict(X_test)
        
        # Generate report
        report = classification_report(
            y_test, y_pred,
            target_names=self.CATEGORIES,
            output_dict=True,
            zero_division=0,
        )
        
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "test_accuracy": report["accuracy"],
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        
        self.is_trained = True
        
        logger.info(f"Training complete:")
        logger.info(f"  CV accuracy: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
        logger.info(f"  Test accuracy: {metrics['test_accuracy']:.3f}")
        
        return metrics

    def predict(
        self,
        features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Predict eligibility category.
        
        Args:
            features: Dict with feature names and values
            
        Returns:
            Dict with prediction results
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        
        # Prepare features
        X = self.prepare_features(features)
        
        # Predict
        category_idx = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Get feature importances
        importances = self.get_feature_importances()
        
        result = {
            "category": self.CATEGORIES[category_idx],
            "confidence": float(probabilities[category_idx]),
            "probabilities": {
                cat: float(prob)
                for cat, prob in zip(self.CATEGORIES, probabilities)
            },
            "feature_importances": importances,
        }
        
        logger.debug(f"Prediction: {result['category']} (confidence: {result['confidence']:.3f})")
        
        return result

    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importances from trained model.
        
        Returns:
            Dict mapping feature names to importance scores
        """
        if not self.is_trained:
            return {}
        
        # HistGradientBoostingClassifier doesn't have feature_importances_
        # We'll use permutation importance or return empty dict for now
        try:
            importances = self.model.feature_importances_
        except AttributeError:
            # HistGradientBoostingClassifier doesn't expose feature_importances_
            # Return uniform importances as a fallback
            logger.warning("Model doesn't support feature_importances_, using uniform weights")
            importances = [1.0 / len(self.FEATURE_NAMES)] * len(self.FEATURE_NAMES)
        
        # Sort by importance
        importance_dict = {
            name: float(imp)
            for name, imp in zip(self.FEATURE_NAMES, importances)
        }
        
        # Sort descending
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        return importance_dict

    def save(self, path: Optional[Path] = None):
        """Save trained model to disk.
        
        Args:
            path: Path to save model (optional, uses default if not provided)
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        save_path = path or self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info(f"Model saved to: {save_path}")

    def load(self, path: Optional[Path] = None):
        """Load trained model from disk.
        
        Args:
            path: Path to load model from (optional, uses default if not provided)
        """
        load_path = path or self.model_path
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            self.model = pickle.load(f)
        
        self.is_trained = True
        
        logger.info(f"Model loaded from: {load_path}")


# Global instance
_classifier = None


def get_classifier() -> EligibilityClassifier:
    """Get or create global classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = EligibilityClassifier()
        # Try to load existing model
        if _classifier.model_path.exists():
            try:
                _classifier.load()
                logger.info("Loaded pre-trained classifier")
            except Exception as e:
                logger.warning(f"Failed to load classifier: {e}")
    return _classifier


if __name__ == "__main__":
    # Test classifier with dummy data
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy training data
    np.random.seed(42)
    n_samples = 100
    
    X = np.random.randn(n_samples, len(EligibilityClassifier.FEATURE_NAMES))
    y = np.random.choice(3, n_samples)  # 0, 1, 2 for three categories
    
    # Train
    classifier = EligibilityClassifier()
    metrics = classifier.train(X, y)
    
    print(f"\n✓ Classifier trained")
    print(f"  CV accuracy: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
    
    # Test prediction
    test_features = {
        name: float(np.random.randn())
        for name in EligibilityClassifier.FEATURE_NAMES
    }
    
    result = classifier.predict(test_features)
    print(f"\n✓ Test prediction: {result['category']} ({result['confidence']:.3f})")
    print(f"  Top features: {list(result['feature_importances'].keys())[:3]}")
