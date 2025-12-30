"""DecisionTreeClassifier model definition and utilities."""
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
import numpy as np


def create_model(config):
    """Create DecisionTreeClassifier model from configuration."""
    train_config = config['train']['config']
    
    model = DecisionTreeClassifier(
        criterion=train_config['criterion'],
        max_depth=train_config['max_depth'],
        min_samples_split=train_config['min_samples_split'],
        min_samples_leaf=train_config['min_samples_leaf'],
        max_features=train_config['max_features'],
        max_leaf_nodes=train_config['max_leaf_nodes'],
        random_state=train_config['random_state'],
        ccp_alpha=train_config['ccp_alpha']
    )
    
    return model


def get_feature_importance(model, feature_names, top_n=20):
    """Get feature importance from trained model."""
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importance = model.feature_importances_
    
    # Get top features
    indices = np.argsort(importance)[-top_n:][::-1]
    top_features = [(feature_names[i], importance[i]) for i in indices]
    
    return top_features


def export_tree_structure(model, feature_names, output_path):
    """Export tree structure to text file."""
    tree_text = export_text(model, feature_names=feature_names, max_depth=10)
    
    with open(output_path, 'w') as f:
        f.write(tree_text)
    
    return tree_text
