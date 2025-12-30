"""SGDClassifier model definition and utilities."""
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np


def create_model(config):
    """Create SGDClassifier model from configuration."""
    train_config = config['train']['config']
    
    model = SGDClassifier(
        loss=train_config['loss'],
        penalty=train_config['penalty'],
        alpha=train_config['alpha'],
        max_iter=train_config['epochs'],
        learning_rate=train_config['learning_rate'],
        early_stopping=train_config['early_stopping'],
        validation_fraction=train_config['validation_fraction'],
        n_iter_no_change=train_config['n_iter_no_change'],
        random_state=train_config['random_state'],
        n_jobs=train_config['threads'],
        verbose=1
    )
    
    return model


def get_feature_importance(model, feature_names, top_n=20):
    """Get feature importance from trained model."""
    if not hasattr(model, 'coef_'):
        return None
        
    # For multiclass, average absolute coefficients across classes
    if len(model.coef_.shape) > 1:
        importance = np.abs(model.coef_).mean(axis=0)
    else:
        importance = np.abs(model.coef_)
    
    # Get top features
    indices = np.argsort(importance)[-top_n:][::-1]
    top_features = [(feature_names[i], importance[i]) for i in indices]
    
    return top_features
