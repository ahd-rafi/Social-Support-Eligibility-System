"""Train the eligibility classifier on synthetic data."""

import asyncio
import logging
import numpy as np
import pandas as pd

from src.db.postgres import PostgresClient
from src.ml.classifier import EligibilityClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def load_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Load training data from PostgreSQL.
    
    Returns:
        Tuple of (X, y) where X is feature matrix and y is labels
    """
    client = PostgresClient()
    
    try:
        await client.connect()
        
        logger.info("Loading training data from PostgreSQL...")
        
        # Query features and eligibility from seeded data
        query = """
            SELECT 
                f.monthly_income,
                f.employment_months,
                f.household_size,
                f.num_dependents,
                f.income_per_family,
                f.total_assets,
                f.total_liabilities,
                f.net_worth,
                f.debt_to_income,
                f.credit_score,
                f.missed_payments,
                f.age
            FROM applicant_features f
            ORDER BY f.id
        """
        
        features_data = await client.fetch(query)
        
        if not features_data:
            raise ValueError("No training data found. Run 'python -m data.seed' first.")
        
        # Convert to DataFrame
        df = pd.DataFrame(features_data)
        
        # For training, we need labels - these would come from historical decisions
        # For now, we'll generate synthetic labels based on rules
        # In production, this would be actual historical approval/decline data
        
        labels = []
        for _, row in df.iterrows():
            if row['income_per_family'] < 2000 and row['net_worth'] < 50000:
                labels.append("approved_financial")
            elif row['income_per_family'] < 4000:
                labels.append("approved_enablement")
            else:
                labels.append("soft_decline")
        
        # Convert to numpy arrays
        X = df.values
        
        # Convert labels to integers
        category_map = {
            "approved_financial": 0,
            "approved_enablement": 1,
            "soft_decline": 2,
        }
        y = np.array([category_map[label] for label in labels])
        
        logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
        logger.info(f"Label distribution:")
        for cat_name, cat_idx in category_map.items():
            count = np.sum(y == cat_idx)
            pct = (count / len(y)) * 100
            logger.info(f"  {cat_name}: {count} ({pct:.1f}%)")
        
        return X, y
        
    finally:
        await client.disconnect()


async def main():
    """Train and save the classifier."""
    logger.info("=" * 60)
    logger.info("Eligibility Classifier Training")
    logger.info("=" * 60)
    
    # Load training data
    X, y = await load_training_data()
    
    # Create and train classifier
    classifier = EligibilityClassifier()
    
    logger.info("\nTraining classifier...")
    metrics = classifier.train(X, y, cv=5)
    
    # Print detailed results
    logger.info("\n" + "=" * 60)
    logger.info("Training Results")
    logger.info("=" * 60)
    logger.info(f"Dataset: {metrics['n_samples']} samples, {metrics['n_features']} features")
    logger.info(f"Cross-validation: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
    logger.info(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    
    logger.info("\nPer-class metrics:")
    report = metrics['classification_report']
    for category in EligibilityClassifier.CATEGORIES:
        if category in report:
            cat_metrics = report[category]
            logger.info(f"  {category}:")
            logger.info(f"    Precision: {cat_metrics['precision']:.3f}")
            logger.info(f"    Recall: {cat_metrics['recall']:.3f}")
            logger.info(f"    F1-score: {cat_metrics['f1-score']:.3f}")
    
    logger.info("\nFeature importances:")
    importances = classifier.get_feature_importances()
    for i, (feature, importance) in enumerate(importances.items(), 1):
        logger.info(f"  {i}. {feature}: {importance:.4f}")
    
    # Save model
    logger.info("\nSaving model...")
    classifier.save()
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Classifier training complete!")
    logger.info(f"Model saved to: {classifier.model_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
