"""Configuration loader for DecisionTreeModel preprocessing."""
import yaml
import logging
from pathlib import Path
from datetime import datetime


def load_config():
    """Load preprocessing configuration from YAML file."""
    config_path = Path(__file__).parent / "preprocessing.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def set_logger(config):
    """Setup logger based on configuration."""
    log_config = config['preprocessing']['config']['logs']
    log_path = log_config['file_path'].replace('{DATE}', datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    # Create log directory if it doesn't exist
    log_dir = Path(log_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = logging.getLogger('decision_tree_preprocessing')
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_path + 'preprocessing.log')
    fh.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        log_config['base_format'],
        datefmt=log_config['date_format']
    )
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)
    
    return logger
