"""Prediction script for DecisionTreeModel."""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

from .config import load_config


def predict(logger, input_features=None):
    """Make predictions using trained DecisionTree model."""
    config = load_config()
    predict_config = config['predict']
    train_config = config['train']
    
    logger.info("=" * 80)
    logger.info("Starting Decision Tree model prediction")
    logger.info("=" * 80)
    
    # Load model and label encoder
    logger.info("Loading trained model...")
    with open(train_config['artefacts']['model_file'], 'rb') as f:
        model = pickle.load(f)
    
    with open(train_config['artefacts']['label_encoder'], 'rb') as f:
        label_encoder = pickle.load(f)
    
    logger.info("Model loaded successfully")
    
    # Load features if not provided
    if input_features is None:
        logger.info("Loading features from file...")
        input_features = pd.read_csv(predict_config['input']['features_path'], index_col=0)
    
    logger.info(f"Features shape: {input_features.shape}")
    
    # Make predictions
    logger.info("Making predictions...")
    predictions_encoded = model.predict(input_features.values)
    predictions = label_encoder.inverse_transform(predictions_encoded)
    
    # Get probabilities
    logger.info("Computing prediction probabilities...")
    probabilities = model.predict_proba(input_features.values)
    
    # Save predictions
    output_dir = Path(predict_config['output']['predictions_path']).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Saving predictions...")
    predictions_df = pd.DataFrame({
        'prediction': predictions,
        'prediction_encoded': predictions_encoded
    }, index=input_features.index)
    
    predictions_df.to_csv(predict_config['output']['predictions_path'])
    
    proba_df = pd.DataFrame(
        probabilities,
        columns=label_encoder.classes_,
        index=input_features.index
    )
    proba_df.to_csv(predict_config['output']['probabilities_path'])
    logger.info("Probabilities saved")
    
    logger.info("=" * 80)
    logger.info("Prediction completed successfully!")
    logger.info("=" * 80)
    
    return predictions, probabilities
