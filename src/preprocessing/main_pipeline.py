import sys
import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import pickle

# Import optionnel de cv2
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Attention: opencv-python (cv2) n'est pas installé. Les features d'images seront désactivées.")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config_loader import get_config, ensure_output_dirs
from src.logger import build_logger

class DecisionTreePreprocessor:
    """
    Pipeline de preprocessing spécialement adapté pour les arbres de décision.
    Extrait des features numériques simples à partir des données textuelles et d'images.
    """

    def __init__(self):
        self.config = get_config()
        self.preprocessing_config = self.config["PREPROCESSING"]
        self.dt_config = self.config["DECISION_TREE"]
        self.log_config = self.config["LOGS"]

        # Initialiser le logger
        self.logger = build_logger(
            name="dt_preprocessing",
            filepath=self.log_config["filePath"],
            baseformat=self.log_config["baseFormat"],
            dateformat=self.log_config["dateFormat"],
            level=logging.INFO
        )

        # Initialiser les transformateurs
        self.tfidf_vectorizer = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def load_data(self):
        """Charge les données brutes"""
        self.logger.info("Chargement des données...")

        # Charger les données textuelles
        text_path = self.preprocessing_config["paths"]["rawTextData"]
        self.X_text = pd.read_csv(text_path)
        self.logger.info(f"Données textuelles chargées: {self.X_text.shape}")

        # Charger les labels
        labels_path = self.preprocessing_config["paths"]["rawLabels"]
        self.y = pd.read_csv(labels_path)
        self.logger.info(f"Labels chargés: {self.y.shape}")

        # Chemin des images
        self.images_path = Path(self.preprocessing_config["paths"]["rawImageFolder"])

    def extract_text_features(self):
        """Extrait des features numériques à partir des données textuelles"""
        self.logger.info("Extraction des features textuelles...")

        text_features = []

        # Combiner les colonnes textuelles si elles existent
        text_columns = self.preprocessing_config["pipeline"]["textpipeline"]["constants"]["textualColumns"]

        # Vérifier quelles colonnes existent réellement
        available_columns = [col for col in text_columns if col in self.X_text.columns]

        if available_columns:
            # Combiner les textes
            combined_text = self.X_text[available_columns].fillna('').agg(' '.join, axis=1)

            # Features TF-IDF si activé
            if self.dt_config["feature_engineering"]["text_features"]["use_tfidf"]:
                max_features = self.dt_config["feature_engineering"]["text_features"]["max_features"]
                ngram_range = tuple(self.dt_config["feature_engineering"]["text_features"]["ngram_range"])

                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    stop_words='english'
                )

                tfidf_features = self.tfidf_vectorizer.fit_transform(combined_text).toarray()
                text_features.append(tfidf_features)
                self.logger.info(f"Features TF-IDF extraites: {tfidf_features.shape}")

            # Features statistiques basiques sur le texte
            text_stats = pd.DataFrame({
                'text_length': combined_text.str.len(),
                'word_count': combined_text.str.split().str.len(),
                'char_count_no_spaces': combined_text.str.replace(' ', '').str.len(),
                'punctuation_count': combined_text.str.count(r'[^\w\s]'),
                'uppercase_count': combined_text.str.count(r'[A-Z]'),
                'digit_count': combined_text.str.count(r'\d')
            }).fillna(0)

            text_features.append(text_stats.values)
            self.logger.info(f"Features statistiques textuelles extraites: {text_stats.shape}")

        # Autres colonnes numériques si elles existent
        numeric_columns = self.X_text.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            numeric_features = self.X_text[numeric_columns].fillna(0).values
            text_features.append(numeric_features)
            self.logger.info(f"Features numériques extraites: {numeric_features.shape}")

        # Combiner toutes les features textuelles
        if text_features:
            self.text_features_array = np.hstack(text_features)
        else:
            self.text_features_array = np.array([]).reshape(len(self.X_text), 0)

        self.logger.info(f"Features textuelles finales: {self.text_features_array.shape}")

    def extract_image_features(self):
        """Extrait des features numériques simples à partir des images"""
        self.logger.info("Extraction des features d'images...")

        if not CV2_AVAILABLE:
            self.logger.warning("cv2 non disponible - création de features d'images par défaut")
            # Créer des features par défaut (zéros)
            feature_size = self._get_image_feature_size()
            self.image_features_array = np.zeros((len(self.X_text), feature_size))
            return

        image_features = []
        image_ids = []

        # Obtenir les IDs d'images (supposant une colonne 'imageid' dans les données textuelles)
        if 'imageid' in self.X_text.columns:
            image_ids = self.X_text['imageid'].values
        else:
            # Essayer de déduire l'ID à partir du nom de fichier d'image
            self.logger.warning("Colonne 'imageid' non trouvée, extraction des IDs depuis les noms de fichiers")
            image_files = list(self.images_path.glob("*.jpg"))
            image_ids = [f.stem for f in image_files[:len(self.X_text)]]

        for img_id in image_ids:
            try:
                # Trouver l'image correspondante
                img_files = list(self.images_path.glob(f"*{img_id}*"))
                if not img_files:
                    # Image par défaut (features nulles)
                    features = np.zeros(self._get_image_feature_size())
                else:
                    img_path = img_files[0]
                    features = self._extract_single_image_features(img_path)

                image_features.append(features)

            except Exception as e:
                self.logger.warning(f"Erreur lors du traitement de l'image {img_id}: {e}")
                # Features par défaut en cas d'erreur
                image_features.append(np.zeros(self._get_image_feature_size()))

        self.image_features_array = np.array(image_features)
        self.logger.info(f"Features d'images extraites: {self.image_features_array.shape}")

    def _extract_single_image_features(self, img_path):
        """Extrait des features d'une seule image"""
        if not CV2_AVAILABLE:
            return np.zeros(self._get_image_feature_size())

        # Charger l'image
        img = cv2.imread(str(img_path))
        if img is None:
            return np.zeros(self._get_image_feature_size())

        features = []

        # Features de base si activées
        if self.dt_config["feature_engineering"]["image_features"]["use_basic_stats"]:
            # Statistiques de base par canal
            for channel in range(3):  # RGB
                channel_data = img[:, :, channel]
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.min(channel_data),
                    np.max(channel_data),
                    np.median(channel_data)
                ])

            # Statistiques globales
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features.extend([
                np.mean(gray),
                np.std(gray),
                img.shape[0],  # hauteur
                img.shape[1],  # largeur
                img.shape[0] * img.shape[1]  # surface
            ])

        # Histogramme si activé
        if self.dt_config["feature_engineering"]["image_features"]["use_histogram"]:
            bins = self.dt_config["feature_engineering"]["image_features"]["histogram_bins"]

            # Histogramme par canal
            for channel in range(3):
                hist, _ = np.histogram(img[:, :, channel], bins=bins, range=(0, 256))
                features.extend(hist)

        return np.array(features)

    def _get_image_feature_size(self):
        """Calcule la taille des features d'image"""
        size = 0

        if self.dt_config["feature_engineering"]["image_features"]["use_basic_stats"]:
            size += 3 * 5 + 5  # 5 stats par canal RGB + 5 stats globales

        if self.dt_config["feature_engineering"]["image_features"]["use_histogram"]:
            bins = self.dt_config["feature_engineering"]["image_features"]["histogram_bins"]
            size += 3 * bins  # histogramme par canal RGB

        return size

    def prepare_labels(self):
        """Prépare les labels pour l'entraînement"""
        self.logger.info("Préparation des labels...")

        # Encoder les labels
        label_column = 'prdtypecode' if 'prdtypecode' in self.y.columns else self.y.columns[0]
        self.labels_encoded = self.label_encoder.fit_transform(self.y[label_column])

        self.logger.info(f"Nombre de classes: {len(self.label_encoder.classes_)}")

    def combine_features(self):
        """Combine toutes les features en un seul dataset"""
        self.logger.info("Combinaison des features...")

        # Combiner les features textuelles et d'images
        features_list = []

        if self.text_features_array.shape[1] > 0:
            features_list.append(self.text_features_array)

        if self.image_features_array.shape[1] > 0:
            features_list.append(self.image_features_array)

        if features_list:
            self.X_combined = np.hstack(features_list)
        else:
            raise ValueError("Aucune feature extraite !")

        self.logger.info(f"Features combinées: {self.X_combined.shape}")

    def split_and_scale_data(self):
        """Divise les données en train/test et applique la normalisation"""
        self.logger.info("Division et normalisation des données...")

        train_size = self.preprocessing_config["pipeline"]["trainSize"]
        random_state = self.preprocessing_config["pipeline"]["randomState"]

        # Division train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_combined,
            self.labels_encoded,
            train_size=train_size,
            random_state=random_state,
            stratify=self.labels_encoded
        )

        # Normalisation
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        self.logger.info(f"Données d'entraînement: {self.X_train_scaled.shape}")
        self.logger.info(f"Données de test: {self.X_test_scaled.shape}")

    def save_preprocessed_data(self):
        """Sauvegarde les données préprocessées"""
        self.logger.info("Sauvegarde des données préprocessées...")

        # Créer les répertoires de sortie
        ensure_output_dirs(self.config)

        # Sauvegarder les features
        features_path = self.dt_config["output_paths"]["features_path"]

        # Créer un DataFrame avec toutes les features
        feature_names = []

        # Noms des features TF-IDF
        if self.tfidf_vectorizer is not None:
            feature_names.extend([f'tfidf_{i}' for i in range(len(self.tfidf_vectorizer.get_feature_names_out()))])

        # Noms des features statistiques textuelles
        feature_names.extend(['text_length', 'word_count', 'char_count_no_spaces',
                             'punctuation_count', 'uppercase_count', 'digit_count'])

        # Noms des features d'images
        if self.dt_config["feature_engineering"]["image_features"]["use_basic_stats"]:
            for channel in ['R', 'G', 'B']:
                feature_names.extend([f'{channel}_mean', f'{channel}_std', f'{channel}_min',
                                    f'{channel}_max', f'{channel}_median'])
            feature_names.extend(['gray_mean', 'gray_std', 'height', 'width', 'area'])

        if self.dt_config["feature_engineering"]["image_features"]["use_histogram"]:
            bins = self.dt_config["feature_engineering"]["image_features"]["histogram_bins"]
            for channel in ['R', 'G', 'B']:
                feature_names.extend([f'{channel}_hist_{i}' for i in range(bins)])

        # Ajuster la longueur des noms de features
        while len(feature_names) < self.X_combined.shape[1]:
            feature_names.append(f'feature_{len(feature_names)}')

        # Créer le DataFrame final
        df_features = pd.DataFrame(self.X_combined, columns=feature_names[:self.X_combined.shape[1]])
        df_features['label'] = self.labels_encoded

        df_features.to_csv(features_path, index=False)
        self.logger.info(f"Features sauvegardées dans: {features_path}")

        # Sauvegarder les transformateurs
        transformers = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_names': feature_names
        }

        transformers_path = features_path.replace('.csv', '_transformers.pkl')
        with open(transformers_path, 'wb') as f:
            pickle.dump(transformers, f)

        self.logger.info(f"Transformateurs sauvegardés dans: {transformers_path}")

    def run_preprocessing(self):
        """Exécute le pipeline complet de preprocessing"""
        self.logger.info("Démarrage du preprocessing pour arbres de décision...")

        try:
            self.load_data()
            self.extract_text_features()
            self.extract_image_features()
            self.prepare_labels()
            self.combine_features()
            self.split_and_scale_data()
            self.save_preprocessed_data()

            self.logger.info("Preprocessing terminé avec succès !")

            # Retourner les données préparées
            return {
                'X_train': self.X_train_scaled,
                'X_test': self.X_test_scaled,
                'y_train': self.y_train,
                'y_test': self.y_test,
                'feature_names': list(range(self.X_combined.shape[1])),
                'label_encoder': self.label_encoder
            }

        except Exception as e:
            self.logger.error(f"Erreur lors du preprocessing: {e}")
            raise

def main():
    """Fonction principale pour lancer le preprocessing"""
    preprocessor = DecisionTreePreprocessor()
    return preprocessor.run_preprocessing()

if __name__ == "__main__":
    main()
