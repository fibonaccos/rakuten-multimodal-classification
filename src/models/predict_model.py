import sys
import os
import logging
import joblib
import pandas as pd
import numpy as np
from typing import Union, List

# Ajout du chemin pour importer les modules src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config_loader import get_config
from src.logger import build_logger

# Configuration
TRAINING_CONFIG = get_config("TRAINING")["SGD"]
LOG_CONFIG = get_config("LOGS")

# Logger
logger = build_logger(
    name="sgd_prediction",
    filepath=LOG_CONFIG["filePath"] + "prediction.log",
    baseformat=LOG_CONFIG["baseFormat"],
    dateformat=LOG_CONFIG["dateFormat"],
    level=logging.INFO
)


class SGDPredictor:
    """Classe pour effectuer des prédictions avec le modèle SGD entraîné"""

    def __init__(self, model_path: str = None):
        """
        Initialise le prédicteur

        Args:
            model_path: Chemin vers le modèle sauvegardé. Si None, utilise le chemin de config
        """
        self.model_path = model_path or TRAINING_CONFIG["PATHS"]["modelSave"]
        self.model = None
        self.label_encoder = None
        self.is_loaded = False

    def load_model(self):
        """Charge le modèle et le label encoder"""
        try:
            # Chargement du modèle
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")

            self.model = joblib.load(self.model_path)
            logger.info(f"Modèle chargé: {self.model_path}")

            # Chargement du label encoder si disponible
            encoder_path = self.model_path.replace('.pkl', '_label_encoder.pkl')
            if os.path.exists(encoder_path):
                self.label_encoder = joblib.load(encoder_path)
                logger.info(f"Label encoder chargé: {encoder_path}")

            self.is_loaded = True
            logger.info("Modèle prêt pour les prédictions")

        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Effectue des prédictions

        Args:
            X: Features pour la prédiction

        Returns:
            Prédictions sous forme de labels originaux si label_encoder disponible
        """
        if not self.is_loaded:
            self.load_model()

        # Prédictions
        predictions = self.model.predict(X)

        # Décodage des labels si nécessaire
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)

        logger.info(f"Prédictions effectuées pour {len(predictions)} échantillons")
        return predictions

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Effectue des prédictions avec probabilités (si supporté par le modèle)

        Args:
            X: Features pour la prédiction

        Returns:
            Probabilités des classes
        """
        if not self.is_loaded:
            self.load_model()

        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            logger.info(f"Probabilités calculées pour {len(probabilities)} échantillons")
            return probabilities
        else:
            logger.warning("Le modèle ne supporte pas predict_proba")
            # Utilisation de decision_function si disponible
            if hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(X)
                logger.info(f"Scores de décision calculés pour {len(scores)} échantillons")
                return scores
            else:
                raise AttributeError("Le modèle ne supporte ni predict_proba ni decision_function")

    def get_model_info(self) -> dict:
        """Retourne les informations sur le modèle chargé"""
        if not self.is_loaded:
            self.load_model()

        return {
            "model_type": type(self.model).__name__,
            "model_params": self.model.get_params(),
            "classes": getattr(self.model, 'classes_', None),
            "n_features": getattr(self.model, 'n_features_in_', None)
        }


def predict_from_csv(input_csv: str, output_csv: str, model_path: str = None):
    """
    Effectue des prédictions à partir d'un fichier CSV

    Args:
        input_csv: Chemin vers le fichier CSV d'entrée
        output_csv: Chemin vers le fichier CSV de sortie
        model_path: Chemin vers le modèle (optionnel)
    """
    logger.info(f"Début des prédictions pour: {input_csv}")

    # Chargement des données
    data = pd.read_csv(input_csv)
    logger.info(f"Données chargées: {data.shape}")

    # Initialisation du prédicteur
    predictor = SGDPredictor(model_path)

    # Prédictions
    predictions = predictor.predict(data)

    # Création du DataFrame de résultats
    results = data.copy()
    results['prediction'] = predictions

    # Ajout des probabilités si possible
    try:
        probabilities = predictor.predict_proba(data)
        if probabilities.ndim == 2:
            # Multi-classe
            for i in range(probabilities.shape[1]):
                results[f'proba_class_{i}'] = probabilities[:, i]
        else:
            # Binaire ou scores de décision
            results['decision_score'] = probabilities
    except Exception as e:
        logger.warning(f"Impossible de calculer les probabilités: {str(e)}")

    # Sauvegarde
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results.to_csv(output_csv, index=False)
    logger.info(f"Prédictions sauvegardées: {output_csv}")


if __name__ == "__main__":
    # Exemple d'utilisation
    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        model_path = sys.argv[3] if len(sys.argv) > 3 else None

        predict_from_csv(input_file, output_file, model_path)
    else:
        print("Usage: python predict_model.py <input_csv> <output_csv> [model_path]")
        print("Exemple: python predict_model.py data/test.csv results/predictions.csv")
