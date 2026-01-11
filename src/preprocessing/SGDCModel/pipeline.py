"""Preprocessing pipeline for SGDCModel."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

from .config import load_config
from .components import TextCleaner, TextVectorizer, ImageFeatureExtractor


def pipe(logger):
    """Execute the complete preprocessing pipeline."""
    config = load_config()
    preproc_config = config['preprocessing']
    
    logger.info("=" * 80)
    logger.info("Starting SGDC preprocessing pipeline")
    logger.info("=" * 80)
    
    # Load data
    logger.info("Loading raw data...")
    X = pd.read_csv(preproc_config['input']['text_path'], index_col=0)
    y = pd.read_csv(preproc_config['input']['labels_path'], index_col=0)
    
    sample_size = preproc_config['config']['sample_size']
    if sample_size > 0:
        X = X.head(sample_size)
        y = y.head(sample_size)
        logger.info(f"Using sample size: {sample_size}")
    
    logger.info(f"Data loaded: X shape {X.shape}, y shape {y.shape}")
    
    # Split data
    logger.info("Splitting data into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=preproc_config['config']['train_size'],
        random_state=preproc_config['config']['random_state'],
        stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Save intermediate labels
    output_dir = Path(preproc_config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    y_train.to_csv(preproc_config['output']['train_labels'])
    y_test.to_csv(preproc_config['output']['test_labels'])
    logger.info("Labels saved")
    
    # Text preprocessing
    steps_config = preproc_config['steps']
    
    if steps_config['text_cleaning']['enable']:
        logger.info("Cleaning text...")
        cleaner = TextCleaner(
            lowercase=steps_config['text_cleaning']['lowercase'],
            remove_punctuation=steps_config['text_cleaning']['remove_punctuation'],
            remove_stopwords=steps_config['text_cleaning']['remove_stopwords']
        )
        X_train = cleaner.fit_transform(X_train)
        X_test = cleaner.transform(X_test)
        logger.info("Text cleaning completed")
    
    # Text vectorization
    if steps_config['tfidf_vectorization']['enable']:
        logger.info("Vectorizing text with TF-IDF...")
        vectorizer = TextVectorizer(
            max_features=steps_config['tfidf_vectorization']['max_features'],
            ngram_range=tuple(steps_config['tfidf_vectorization']['ngram_range']),
            min_df=steps_config['tfidf_vectorization']['min_df'],
            max_df=steps_config['tfidf_vectorization']['max_df']
        )
        X_train_text = vectorizer.fit_transform(X_train)
        X_test_text = vectorizer.transform(X_test)
        logger.info(f"Text vectorization completed: {X_train_text.shape[1]} features")
    else:
        X_train_text = pd.DataFrame(index=X_train.index)
        X_test_text = pd.DataFrame(index=X_test.index)
    
    # Image features
    if steps_config['image_features']['enable']:
        logger.info("Extracting image features...")
        img_extractor = ImageFeatureExtractor(
            image_dir=preproc_config['input']['image_dir'],
            extract_color_histograms=steps_config['image_features']['extract_color_histograms'],
            resize_shape=tuple(steps_config['image_features']['resize_shape'])
        )
        X_train_img = img_extractor.fit_transform(X_train)
        X_test_img = img_extractor.transform(X_test)
        logger.info(f"Image feature extraction completed: {X_train_img.shape[1]} features")
    else:
        X_train_img = pd.DataFrame(index=X_train.index)
        X_test_img = pd.DataFrame(index=X_test.index)
    
    # Combine features
    logger.info("Combining all features...")
    X_train_combined = pd.concat([X_train_text, X_train_img], axis=1)
    X_test_combined = pd.concat([X_test_text, X_test_img], axis=1)
    logger.info(f"Combined features shape - Train: {X_train_combined.shape}, Test: {X_test_combined.shape}")
    
    # Save features
    logger.info("Saving processed features...")
    X_train_combined.to_csv(preproc_config['output']['train_features'], index=True)
    X_test_combined.to_csv(preproc_config['output']['test_features'], index=True)
    
    # Save transformers
    transformers = {
        'text_cleaner': cleaner if steps_config['text_cleaning']['enable'] else None,
        'text_vectorizer': vectorizer if steps_config['tfidf_vectorization']['enable'] else None,
        'image_extractor': img_extractor if steps_config['image_features']['enable'] else None,
        'feature_names': X_train_combined.columns.tolist()
    }
    
    with open(preproc_config['output']['transformers'], 'wb') as f:
        pickle.dump(transformers, f)
    
    logger.info("Transformers saved")
    logger.info("=" * 80)
    logger.info("SGDC preprocessing pipeline completed successfully")
    logger.info("=" * 80)
    
    return {
        'X_train': X_train_combined,
        'X_test': X_test_combined,
        'y_train': y_train,
        'y_test': y_test
    }
