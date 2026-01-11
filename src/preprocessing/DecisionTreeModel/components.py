"""Preprocessing components for DecisionTreeModel."""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re
from pathlib import Path
import pickle


class TextCleaner(BaseEstimator, TransformerMixin):
    """Clean and preprocess text data."""
    
    def __init__(self, lowercase=True, remove_punctuation=True, remove_stopwords=True):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.stopwords = set(['le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'ou'])
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        """Transform text data."""
        X_copy = X.copy()
        text_cols = ['designation', 'description']
        
        for col in text_cols:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].apply(self._clean_text)
                
        return X_copy
    
    def _clean_text(self, text):
        """Clean individual text."""
        if pd.isna(text):
            return ""
            
        text = str(text)
        
        if self.lowercase:
            text = text.lower()
            
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
            
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        if self.remove_stopwords:
            words = text.split()
            text = ' '.join([w for w in words if w not in self.stopwords])
            
        return text


class TextVectorizer(BaseEstimator, TransformerMixin):
    """Vectorize text using TF-IDF."""
    
    def __init__(self, max_features=3000, ngram_range=(1, 2), min_df=3, max_df=0.9):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizers = {}
        
    def fit(self, X, y=None):
        """Fit TF-IDF vectorizers on text columns."""
        text_cols = ['designation', 'description']
        
        for col in text_cols:
            if col in X.columns:
                vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                    min_df=self.min_df,
                    max_df=self.max_df
                )
                vectorizer.fit(X[col].fillna(''))
                self.vectorizers[col] = vectorizer
                
        return self
        
    def transform(self, X):
        """Transform text columns to TF-IDF features."""
        features_list = []
        feature_names = []
        
        for col, vectorizer in self.vectorizers.items():
            if col in X.columns:
                vectors = vectorizer.transform(X[col].fillna('')).toarray()
                features_list.append(vectors)
                feature_names.extend([f"{col}_{i}" for i in range(vectors.shape[1])])
                
        if features_list:
            all_features = np.hstack(features_list)
            result_df = pd.DataFrame(all_features, columns=feature_names, index=X.index)
            return result_df
        else:
            return pd.DataFrame(index=X.index)


class ImageFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract basic features from images."""
    
    def __init__(self, image_dir, extract_color_histograms=True, resize_shape=(128, 128)):
        self.image_dir = Path(image_dir)
        self.extract_color_histograms = extract_color_histograms
        self.resize_shape = resize_shape
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        """Extract features from images."""
        features_list = []
        
        for idx, row in X.iterrows():
            try:
                image_id = row['imageid']
                product_id = row['productid']
                image_path = self.image_dir / f"image_{image_id}_product_{product_id}.jpg"
                
                if self.extract_color_histograms:
                    features = self._extract_color_histogram(image_path)
                else:
                    features = [0] * 192  # Default empty features
                    
                features_list.append(features)
            except Exception as e:
                # Default features if image not found
                features_list.append([0] * 192)
                
        feature_names = [f"img_hist_{i}" for i in range(len(features_list[0]))]
        result_df = pd.DataFrame(features_list, columns=feature_names, index=X.index)
        return result_df
    
    def _extract_color_histogram(self, image_path):
        """Extract color histogram from image."""
        try:
            from PIL import Image
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.resize_shape)
            
            # Compute histograms for each channel
            img_array = np.array(img)
            hist_r = np.histogram(img_array[:,:,0], bins=64, range=(0, 256))[0]
            hist_g = np.histogram(img_array[:,:,1], bins=64, range=(0, 256))[0]
            hist_b = np.histogram(img_array[:,:,2], bins=64, range=(0, 256))[0]
            
            # Normalize
            hist_r = hist_r / hist_r.sum()
            hist_g = hist_g / hist_g.sum()
            hist_b = hist_b / hist_b.sum()
            
            return np.concatenate([hist_r, hist_g, hist_b]).tolist()
        except:
            return [0] * 192
