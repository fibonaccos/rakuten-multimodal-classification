import sys
import os
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config_loader import get_config, ensure_output_dirs
from src.logger import build_logger
from src.preprocessing.main_pipeline import DecisionTreePreprocessor

class DecisionTreeTrainer:
    """
    Classe pour entraîner et évaluer des modèles DecisionTreeClassifier
    """

    def __init__(self):
        self.config = get_config()
        self.dt_config = self.config["DECISION_TREE"]
        self.log_config = self.config["LOGS"]

        # Initialiser le logger
        log_path = self.log_config["filePath"].replace("preprocessing", "training")
        self.logger = build_logger(
            name="dt_training",
            filepath=log_path,
            baseformat=self.log_config["baseFormat"],
            dateformat=self.log_config["dateFormat"],
            level=logging.INFO
        )

        # Initialiser le modèle
        self.model = None
        self.best_model = None

    def load_preprocessed_data(self):
        """Charge les données préprocessées ou lance le preprocessing"""
        features_path = self.dt_config["output_paths"]["features_path"]

        if os.path.exists(features_path):
            self.logger.info("Chargement des données préprocessées...")

            # Charger les données
            df = pd.read_csv(features_path)
            self.X = df.drop('label', axis=1).values
            self.y = df['label'].values

            # Charger les transformateurs
            transformers_path = features_path.replace('.csv', '_transformers.pkl')
            with open(transformers_path, 'rb') as f:
                transformers = pickle.load(f)

            self.label_encoder = transformers['label_encoder']
            self.feature_names = transformers['feature_names']

            self.logger.info(f"Données chargées: {self.X.shape}")

        else:
            self.logger.info("Données préprocessées non trouvées, lancement du preprocessing...")

            # Lancer le preprocessing
            preprocessor = DecisionTreePreprocessor()
            data = preprocessor.run_preprocessing()

            self.X = np.vstack([data['X_train'], data['X_test']])
            self.y = np.concatenate([data['y_train'], data['y_test']])
            self.label_encoder = data['label_encoder']
            self.feature_names = data['feature_names']

    def split_data(self):
        """Divise les données en ensembles d'entraînement et de test"""
        from sklearn.model_selection import train_test_split

        train_size = self.config["PREPROCESSING"]["pipeline"]["trainSize"]
        random_state = self.config["PREPROCESSING"]["pipeline"]["randomState"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            train_size=train_size,
            random_state=random_state,
            stratify=self.y
        )

        self.logger.info(f"Division des données - Train: {self.X_train.shape}, Test: {self.X_test.shape}")

    def train_basic_model(self):
        """Entraîne un modèle de base avec les paramètres de configuration"""
        self.logger.info("Entraînement du modèle de base...")

        # Récupérer les paramètres du modèle
        model_params = self.dt_config["model_params"]

        # Créer et entraîner le modèle
        self.model = DecisionTreeClassifier(**model_params)
        self.model.fit(self.X_train, self.y_train)

        # Évaluation sur l'ensemble de test
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)

        self.logger.info(f"Précision du modèle de base: {accuracy:.4f}")

        return accuracy

    def hyperparameter_tuning(self):
        """Optimise les hyperparamètres avec GridSearchCV"""
        self.logger.info("Optimisation des hyperparamètres...")

        # Grille de paramètres à tester
        param_grid = {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2', None]
        }

        # GridSearchCV
        dt = DecisionTreeClassifier(random_state=self.dt_config["model_params"]["random_state"])
        grid_search = GridSearchCV(
            dt,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(self.X_train, self.y_train)

        # Meilleur modèle
        self.best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_

        self.logger.info(f"Meilleurs paramètres: {grid_search.best_params_}")
        self.logger.info(f"Meilleur score CV: {best_score:.4f}")

        # Évaluation sur l'ensemble de test
        y_pred_best = self.best_model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_pred_best)

        self.logger.info(f"Précision du meilleur modèle sur le test: {test_accuracy:.4f}")

        return test_accuracy, grid_search.best_params_

    def evaluate_model(self, model=None):
        """Évalue le modèle de manière détaillée"""
        if model is None:
            model = self.best_model if self.best_model is not None else self.model

        self.logger.info("Évaluation détaillée du modèle...")

        # Prédictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None

        # Métriques
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, output_dict=True)
        cm = confusion_matrix(self.y_test, y_pred)

        # Validation croisée
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')

        # Importance des features
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_

        evaluation_results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance.tolist() if feature_importance is not None else None
        }

        self.logger.info(f"Précision finale: {accuracy:.4f}")
        self.logger.info(f"Score CV moyen: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return evaluation_results

    def analyze_feature_importance(self, model=None, top_n=20):
        """Analyse l'importance des features"""
        if model is None:
            model = self.best_model if self.best_model is not None else self.model

        if not hasattr(model, 'feature_importances_'):
            self.logger.warning("Le modèle ne supporte pas l'analyse d'importance des features")
            return None

        self.logger.info("Analyse de l'importance des features...")

        importances = model.feature_importances_

        # Créer un DataFrame avec les importances
        if len(self.feature_names) == len(importances):
            feature_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            })
        else:
            feature_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importances))],
                'importance': importances
            })

        # Trier par importance
        feature_df = feature_df.sort_values('importance', ascending=False)

        # Afficher les top features
        self.logger.info(f"Top {top_n} features les plus importantes:")
        for idx, row in feature_df.head(top_n).iterrows():
            self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return feature_df

    def save_model_and_results(self, evaluation_results, feature_importance_df=None):
        """Sauvegarde le modèle et les résultats"""
        self.logger.info("Sauvegarde du modèle et des résultats...")

        # Créer les répertoires de sortie
        ensure_output_dirs(self.config)

        # Sauvegarder le modèle
        model_path = self.dt_config["output_paths"]["model_path"]
        model_to_save = self.best_model if self.best_model is not None else self.model

        joblib.dump(model_to_save, model_path)
        self.logger.info(f"Modèle sauvegardé dans: {model_path}")

        # Sauvegarder les résultats d'évaluation
        evaluation_path = self.dt_config["output_paths"]["evaluation_path"]

        # Ajouter des métadonnées
        results_with_metadata = {
            'model_type': 'DecisionTreeClassifier',
            'model_parameters': model_to_save.get_params(),
            'feature_count': self.X.shape[1],
            'sample_count': self.X.shape[0],
            'class_count': len(np.unique(self.y)),
            'evaluation': evaluation_results
        }

        with open(evaluation_path, 'w', encoding='utf-8') as f:
            json.dump(results_with_metadata, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Résultats d'évaluation sauvegardés dans: {evaluation_path}")

        # Sauvegarder l'importance des features si disponible
        if feature_importance_df is not None:
            importance_path = evaluation_path.replace('.json', '_feature_importance.csv')
            feature_importance_df.to_csv(importance_path, index=False)
            self.logger.info(f"Importance des features sauvegardée dans: {importance_path}")

    def run_complete_training(self):
        """Exécute le pipeline complet d'entraînement"""
        self.logger.info("Démarrage de l'entraînement complet des arbres de décision...")

        try:
            # Charger les données
            self.load_preprocessed_data()

            # Diviser les données
            self.split_data()

            # Entraîner le modèle de base
            basic_accuracy = self.train_basic_model()

            # Optimiser les hyperparamètres
            tuned_accuracy, best_params = self.hyperparameter_tuning()

            # Évaluer le meilleur modèle
            evaluation_results = self.evaluate_model()

            # Analyser l'importance des features
            feature_importance_df = self.analyze_feature_importance()

            # Sauvegarder tout
            self.save_model_and_results(evaluation_results, feature_importance_df)

            self.logger.info("Entraînement terminé avec succès !")

            return {
                'basic_accuracy': basic_accuracy,
                'tuned_accuracy': tuned_accuracy,
                'best_params': best_params,
                'evaluation': evaluation_results,
                'model': self.best_model
            }

        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement: {e}")
            raise

def main():
    """Fonction principale pour lancer l'entraînement"""
    trainer = DecisionTreeTrainer()
    return trainer.run_complete_training()

if __name__ == "__main__":
    main()
