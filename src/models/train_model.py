import sys
import os
import logging
import json
import pickle
import time
from pathlib import Path

# Ajout du chemin pour importer les modules src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config_loader import get_config
from src.logger import build_logger
from src.utils import timer, format_duration

# Imports pour le machine learning
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix,
                           f1_score, accuracy_score, precision_score, recall_score)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Import conditionnel pour l'optimisation bayésienne
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical
    BAYES_AVAILABLE = True
except ImportError:
    print("Warning: scikit-optimize not available. Bayesian optimization will be disabled.")
    BAYES_AVAILABLE = False

# Configuration
TRAINING_CONFIG = get_config("TRAINING")["SGD"]
LOG_CONFIG = get_config("LOGS")

# Logger
logger = build_logger(
    name="sgd_training",
    filepath=LOG_CONFIG["filePath"] + "training.log",
    baseformat=LOG_CONFIG["baseFormat"],
    dateformat=LOG_CONFIG["dateFormat"],
    level=logging.INFO
)


class SGDTrainer:
    """Classe pour l'entraînement et l'évaluation du modèle SGDClassifier"""

    def __init__(self):
        self.config = TRAINING_CONFIG
        self.model = None
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = None
        self.results = {}

    def load_data(self):
        """Charge les données préprocessées"""
        logger.info("Chargement des données préprocessées...")

        # Chargement des features
        data_path = self.config["PATHS"]["processedFeatures"]
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Fichier de données non trouvé: {data_path}")

        # Chargement par chunks pour gérer les gros fichiers
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Données chargées: {df.shape}")
        except MemoryError:
            logger.warning("Fichier trop volumineux, chargement par chunks...")
            chunks = []
            for chunk in pd.read_csv(data_path, chunksize=10000):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Données chargées par chunks: {df.shape}")

        # Séparation des features et des labels
        # D'après vos logs, la dernière colonne devrait contenir les labels (prdtypecode)
        if 'prdtypecode' in df.columns:
            X = df.drop('prdtypecode', axis=1)
            y = df['prdtypecode']
            logger.info("Colonne 'prdtypecode' trouvée et utilisée comme target")
        elif 'target' in df.columns:
            X = df.drop('target', axis=1)
            y = df['target']
            logger.info("Colonne 'target' trouvée et utilisée comme target")
        else:
            # Si pas de colonne identifiée, chercher la dernière colonne non-numérique
            # ou utiliser la dernière colonne
            potential_targets = []
            for col in df.columns:
                if df[col].dtype == 'object' or df[col].nunique() < 50:  # Potentiel target
                    potential_targets.append(col)

            if potential_targets:
                target_col = potential_targets[-1]  # Prendre le dernier trouvé
                X = df.drop(target_col, axis=1)
                y = df[target_col]
                logger.info(f"Colonne '{target_col}' utilisée comme target")
            else:
                # Dernière colonne par défaut
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
                logger.info("Dernière colonne utilisée comme target")

        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Labels shape: {y.shape}")
        logger.info(f"Nombre de classes uniques: {len(y.unique())}")
        logger.info(f"Classes: {sorted(y.unique())}")

        # Vérification des NaN
        if X.isnull().sum().sum() > 0:
            logger.warning(f"NaN détectés dans les features: {X.isnull().sum().sum()}")
            X = X.fillna(0)  # Remplacer par 0 pour SGD
            logger.info("NaN remplacés par 0")

        if y.isnull().sum() > 0:
            logger.warning(f"NaN détectés dans les labels: {y.isnull().sum()}")
            # Supprimer les lignes avec des labels NaN
            mask = ~y.isnull()
            X = X[mask]
            y = y[mask]
            logger.info(f"Lignes avec labels NaN supprimées. Nouvelle shape: {X.shape}")

        # Encodage des labels si nécessaire
        if y.dtype == 'object' or not np.issubdtype(y.dtype, np.integer):
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            logger.info("Labels encodés avec LabelEncoder")
            logger.info(f"Mapping des classes: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
            y = y_encoded

        # Division train/test
        test_size = self.config["EVALUATION"]["test_size"]
        random_state = self.config["EVALUATION"]["random_state"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y, shuffle=True
        )

        logger.info(f"Données d'entraînement: {self.X_train.shape}")
        logger.info(f"Données de test: {self.X_test.shape}")
        logger.info(f"Distribution des classes en entraînement: {np.bincount(self.y_train)}")
        logger.info(f"Distribution des classes en test: {np.bincount(self.y_test)}")

    def create_base_model(self):
        """Crée le modèle SGD de base"""
        model_params = self.config["MODEL_PARAMS"].copy()
        self.model = SGDClassifier(**model_params)
        logger.info(f"Modèle SGD créé avec les paramètres: {model_params}")

    def perform_grid_search(self):
        """Effectue la recherche par grille"""
        if not self.config["GRID_SEARCH"]["active"]:
            logger.info("Grid Search désactivé")
            return

        logger.info("Début de Grid Search...")
        start_time = time.time()

        param_grid = self.config["GRID_SEARCH"]["param_grid"]
        cv = self.config["GRID_SEARCH"]["cv"]
        scoring = self.config["GRID_SEARCH"]["scoring"]
        n_jobs = self.config["GRID_SEARCH"]["n_jobs"]

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )

        grid_search.fit(self.X_train, self.y_train)

        self.best_model = grid_search.best_estimator_

        end_time = time.time()
        duration = format_duration(end_time - start_time)

        logger.info(f"Grid Search terminé en {duration}")
        logger.info(f"Meilleurs paramètres: {grid_search.best_params_}")
        logger.info(f"Meilleur score CV: {grid_search.best_score_:.4f}")

        # Sauvegarde des résultats de Grid Search
        self.results["grid_search"] = {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": {
                "mean_test_score": grid_search.cv_results_["mean_test_score"].tolist(),
                "std_test_score": grid_search.cv_results_["std_test_score"].tolist(),
                "params": grid_search.cv_results_["params"]
            }
        }

    def perform_bayes_search(self):
        """Effectue la recherche bayésienne"""
        if not self.config["BAYES_SEARCH"]["active"] or not BAYES_AVAILABLE:
            logger.info("Bayes Search désactivé ou non disponible")
            return

        logger.info("Début de Bayesian Search...")
        start_time = time.time()

        # Définition de l'espace de recherche
        search_spaces = {
            'loss': Categorical(self.config["BAYES_SEARCH"]["param_space"]["loss"]),
            'alpha': Real(self.config["BAYES_SEARCH"]["param_space"]["alpha"][0],
                         self.config["BAYES_SEARCH"]["param_space"]["alpha"][1],
                         prior='log-uniform'),
            'learning_rate': Categorical(self.config["BAYES_SEARCH"]["param_space"]["learning_rate"]),
            'eta0': Real(self.config["BAYES_SEARCH"]["param_space"]["eta0"][0],
                        self.config["BAYES_SEARCH"]["param_space"]["eta0"][1],
                        prior='log-uniform')
        }

        bayes_search = BayesSearchCV(
            estimator=self.model,
            search_spaces=search_spaces,
            n_iter=self.config["BAYES_SEARCH"]["n_iter"],
            cv=self.config["BAYES_SEARCH"]["cv"],
            scoring=self.config["BAYES_SEARCH"]["scoring"],
            n_jobs=self.config["BAYES_SEARCH"]["n_jobs"],
            random_state=self.config["BAYES_SEARCH"]["random_state"],
            verbose=1
        )

        bayes_search.fit(self.X_train, self.y_train)

        self.best_model = bayes_search.best_estimator_

        end_time = time.time()
        duration = format_duration(end_time - start_time)

        logger.info(f"Bayesian Search terminé en {duration}")
        logger.info(f"Meilleurs paramètres: {bayes_search.best_params_}")
        logger.info(f"Meilleur score CV: {bayes_search.best_score_:.4f}")

        # Sauvegarde des résultats de Bayesian Search
        self.results["bayes_search"] = {
            "best_params": bayes_search.best_params_,
            "best_score": bayes_search.best_score_
        }

    def train_final_model(self):
        """Entraîne le modèle final"""
        if self.best_model is None:
            logger.info("Entraînement du modèle de base...")
            self.best_model = self.model

        logger.info("Entraînement du modèle final...")
        start_time = time.time()

        self.best_model.fit(self.X_train, self.y_train)

        end_time = time.time()
        duration = format_duration(end_time - start_time)
        logger.info(f"Entraînement terminé en {duration}")

    def evaluate_model(self):
        """Évalue le modèle et génère les métriques"""
        logger.info("Évaluation du modèle...")

        # Prédictions
        y_pred_train = self.best_model.predict(self.X_train)
        y_pred_test = self.best_model.predict(self.X_test)

        # Calcul des métriques
        metrics = {}
        average = self.config["EVALUATION"]["average"]

        # Métriques sur l'ensemble de test
        metrics["test"] = {
            "accuracy": accuracy_score(self.y_test, y_pred_test),
            "f1_score": f1_score(self.y_test, y_pred_test, average=average),
            "precision": precision_score(self.y_test, y_pred_test, average=average),
            "recall": recall_score(self.y_test, y_pred_test, average=average)
        }

        # Métriques sur l'ensemble d'entraînement
        metrics["train"] = {
            "accuracy": accuracy_score(self.y_train, y_pred_train),
            "f1_score": f1_score(self.y_train, y_pred_train, average=average),
            "precision": precision_score(self.y_train, y_pred_train, average=average),
            "recall": recall_score(self.y_train, y_pred_train, average=average)
        }

        self.results["metrics"] = metrics

        # Log des résultats
        logger.info("=== RÉSULTATS D'ÉVALUATION ===")
        logger.info(f"Test Accuracy: {metrics['test']['accuracy']:.4f}")
        logger.info(f"Test F1-Score: {metrics['test']['f1_score']:.4f}")
        logger.info(f"Test Precision: {metrics['test']['precision']:.4f}")
        logger.info(f"Test Recall: {metrics['test']['recall']:.4f}")

        # Rapport de classification
        class_report = classification_report(
            self.y_test, y_pred_test,
            target_names=[str(i) for i in range(len(np.unique(self.y_test)))]
        )

        logger.info("Rapport de classification:")
        logger.info(f"\n{class_report}")

        # Sauvegarde du rapport de classification
        report_path = self.config["PATHS"]["classificationReportSave"]
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            f.write("=== RAPPORT DE CLASSIFICATION SGD ===\n\n")
            f.write(f"Paramètres du modèle: {self.best_model.get_params()}\n\n")
            f.write("=== MÉTRIQUES ===\n")
            f.write(f"Test Accuracy: {metrics['test']['accuracy']:.4f}\n")
            f.write(f"Test F1-Score: {metrics['test']['f1_score']:.4f}\n")
            f.write(f"Test Precision: {metrics['test']['precision']:.4f}\n")
            f.write(f"Test Recall: {metrics['test']['recall']:.4f}\n\n")
            f.write("=== RAPPORT DÉTAILLÉ ===\n")
            f.write(class_report)

        # Matrice de confusion
        self.plot_confusion_matrix(self.y_test, y_pred_test)

        return metrics

    def plot_confusion_matrix(self, y_true, y_pred):
        """Génère et sauvegarde la matrice de confusion"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(len(np.unique(y_true))),
                    yticklabels=range(len(np.unique(y_true))))
        plt.title('Matrice de Confusion - SGD Classifier')
        plt.xlabel('Prédictions')
        plt.ylabel('Vraies valeurs')

        # Sauvegarde
        save_path = self.config["PATHS"]["confusionMatrixSave"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Matrice de confusion sauvegardée: {save_path}")

    def save_model_and_results(self):
        """Sauvegarde le modèle et les résultats"""
        # Sauvegarde du modèle
        model_path = self.config["PATHS"]["modelSave"]
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.best_model, model_path)
        logger.info(f"Modèle sauvegardé: {model_path}")

        # Sauvegarde du label encoder si utilisé
        if self.label_encoder is not None:
            encoder_path = model_path.replace('.pkl', '_label_encoder.pkl')
            joblib.dump(self.label_encoder, encoder_path)
            logger.info(f"Label encoder sauvegardé: {encoder_path}")

        # Sauvegarde des résultats
        results_path = self.config["PATHS"]["resultsSave"]
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        # Ajout des informations du modèle
        self.results["model_info"] = {
            "model_type": "SGDClassifier",
            "model_params": self.best_model.get_params(),
            "feature_count": self.X_train.shape[1],
            "training_samples": self.X_train.shape[0],
            "test_samples": self.X_test.shape[0],
            "n_classes": len(np.unique(self.y_train))
        }

        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Résultats sauvegardés: {results_path}")


@timer
def main():
    """Fonction principale d'entraînement"""
    logger.info("=== DÉBUT DE L'ENTRAÎNEMENT SGD CLASSIFIER ===")

    try:
        # Initialisation
        trainer = SGDTrainer()

        # Chargement des données
        trainer.load_data()

        # Création du modèle de base
        trainer.create_base_model()

        # Optimisation des hyperparamètres
        if trainer.config["GRID_SEARCH"]["active"]:
            trainer.perform_grid_search()
        elif trainer.config["BAYES_SEARCH"]["active"]:
            trainer.perform_bayes_search()

        # Entraînement final
        trainer.train_final_model()

        # Évaluation
        metrics = trainer.evaluate_model()

        # Sauvegarde
        trainer.save_model_and_results()

        logger.info("=== ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS ===")
        logger.info(f"F1-Score final: {metrics['test']['f1_score']:.4f}")
        logger.info(f"Accuracy finale: {metrics['test']['accuracy']:.4f}")

    except Exception as e:
        logger.error(f"Erreur durante l'entraînement: {str(e)}")
        raise


if __name__ == "__main__":
    main()
