"""Training script for SGDCModel."""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from .config import load_config
from .model import create_model, get_feature_importance


def make_dirs():
    """Create necessary directories for artifacts."""
    config = load_config()
    
    dirs = [
        config['train']['artefacts']['base_dir'],
        config['train']['metrics']['base_dir'],
        config['train']['visualization']['base_dir']
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def train_model(logger):
    """Train the SGDC model."""
    config = load_config()
    train_config = config['train']
    
    logger.info("=" * 80)
    logger.info("Starting SGDC model training")
    logger.info("=" * 80)
    
    # Load data
    logger.info("Loading preprocessed data...")
    X_train = pd.read_csv(train_config['data_dir']['train_features'], index_col=0)
    X_test = pd.read_csv(train_config['data_dir']['test_features'], index_col=0)
    y_train = pd.read_csv(train_config['data_dir']['train_labels'], index_col=0).squeeze()
    y_test = pd.read_csv(train_config['data_dir']['test_labels'], index_col=0).squeeze()
    
    logger.info(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Load transformers for feature names
    with open(train_config['data_dir']['transformers'], 'rb') as f:
        transformers = pickle.load(f)
    feature_names = transformers['feature_names']
    
    # Encode labels
    logger.info("Encoding labels...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    logger.info(f"Number of classes: {len(label_encoder.classes_)}")
    
    # Create and train model
    logger.info("Creating model...")
    model = create_model(config)
    
    logger.info("Training model...")
    model.fit(X_train.values, y_train_encoded)
    
    logger.info("Training completed!")
    
    # Evaluate model
    logger.info("Evaluating model on test set...")
    y_pred = model.predict(X_test.values)
    y_pred_proba = model.predict_proba(X_test.values) if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_encoded, y_pred)
    f1_macro = f1_score(y_test_encoded, y_pred, average='macro')
    f1_weighted = f1_score(y_test_encoded, y_pred, average='weighted')
    precision = precision_score(y_test_encoded, y_pred, average='weighted')
    recall = recall_score(y_test_encoded, y_pred, average='weighted')
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 (macro): {f1_macro:.4f}")
    logger.info(f"F1 (weighted): {f1_weighted:.4f}")
    logger.info(f"Precision (weighted): {precision:.4f}")
    logger.info(f"Recall (weighted): {recall:.4f}")
    
    # Save model and artifacts
    logger.info("Saving model and artifacts...")
    
    with open(train_config['artefacts']['model_file'], 'wb') as f:
        pickle.dump(model, f)
    
    with open(train_config['artefacts']['label_encoder'], 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'precision_weighted': float(precision),
        'recall_weighted': float(recall),
        'n_classes': int(len(label_encoder.classes_)),
        'n_features': int(X_train.shape[1])
    }
    
    with open(train_config['metrics']['metrics_summary'], 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Classification report
    report = classification_report(
        y_test_encoded, y_pred,
        labels=range(len(label_encoder.classes_)),
        target_names=label_encoder.classes_.astype(str),
        output_dict=True,
        zero_division=0
    )
    
    with open(train_config['metrics']['classification_report'], 'w') as f:
        json.dump(report, f, indent=2)
    
    # Confusion matrix
    logger.info("Generating confusion matrix...")
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - SGDC Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(train_config['metrics']['confusion_matrix'], dpi=150)
    plt.close()
    
    # Feature importance
    logger.info("Analyzing feature importance...")
    top_features = get_feature_importance(model, feature_names, top_n=20)
    
    if top_features:
        fig, ax = plt.subplots(figsize=(10, 8))
        features, importances = zip(*top_features)
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance (Mean Absolute Coefficient)')
        ax.set_title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig(train_config['visualization']['feature_importance'], dpi=150)
        plt.close()
    
    logger.info("=" * 80)
    logger.info("Training completed successfully!")
    logger.info("=" * 80)
    
    return model, label_encoder, metrics
