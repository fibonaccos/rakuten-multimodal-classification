"""
Script de démonstration pour tester la configuration complète du preprocessing
et de l'entraînement des arbres de décision.
"""

import sys
import os
import logging

# Ajouter le chemin vers le module src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config_loader import get_config, validate_paths, ensure_output_dirs
from src.logger import build_logger

def test_configuration():
    """Test de la configuration complète"""
    print("=" * 60)
    print("TEST DE LA CONFIGURATION PREPROCESSING POUR ARBRES DE DÉCISION")
    print("=" * 60)

    try:
        # Test 1: Chargement de la configuration
        print("\n1. Test du chargement de configuration...")
        config = get_config()
        preprocessing_config = get_config("PREPROCESSING")
        dt_config = get_config("DECISION_TREE")
        logs_config = get_config("LOGS")

        print("✓ Configuration chargée avec succès")
        print(f"  - Sections disponibles: {list(config.keys())}")

        # Test 2: Validation des chemins
        print("\n2. Test de validation des chemins...")
        paths_valid = validate_paths(preprocessing_config)
        if paths_valid:
            print("✓ Tous les chemins de données sont valides")
        else:
            print("⚠ Certains chemins sont manquants (voir logs pour détails)")

        # Test 3: Création des répertoires de sortie
        print("\n3. Test de création des répertoires de sortie...")
        ensure_output_dirs(config)
        print("✓ Répertoires de sortie créés/vérifiés")

        # Test 4: Configuration du logger
        print("\n4. Test de configuration des logs...")
        logger = build_logger(
            name="test_logger",
            filepath=logs_config["filePath"],
            baseformat=logs_config["baseFormat"],
            dateformat=logs_config["dateFormat"],
            level=logging.INFO
        )
        logger.info("Test du système de logs - Configuration réussie !")
        print("✓ Logger configuré et fonctionnel")

        # Test 5: Affichage des paramètres clés
        print("\n5. Paramètres de configuration:")
        print(f"  - Pipeline à exécuter: {preprocessing_config['pipeline']['toPipe']}")
        print(f"  - Taille d'échantillon: {preprocessing_config['pipeline']['sampleSize']}")
        print(f"  - Ratio train/test: {preprocessing_config['pipeline']['trainSize']}")
        print(f"  - Seed aléatoire: {preprocessing_config['pipeline']['randomState']}")

        print(f"  - Modèle DT - Critère: {dt_config['model_params']['criterion']}")
        print(f"  - Modèle DT - Profondeur max: {dt_config['model_params']['max_depth']}")
        print(f"  - Features TF-IDF: {dt_config['feature_engineering']['text_features']['use_tfidf']}")
        print(f"  - Max features TF-IDF: {dt_config['feature_engineering']['text_features']['max_features']}")

        # Test 6: Vérification des dépendances
        print("\n6. Test des dépendances Python...")
        required_packages = [
            'pandas', 'numpy', 'sklearn', 'cv2', 'joblib'
        ]

        missing_packages = []
        for package in required_packages:
            try:
                if package == 'cv2':
                    import cv2
                elif package == 'sklearn':
                    import sklearn
                else:
                    __import__(package)
                print(f"  ✓ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"  ✗ {package} - MANQUANT")

        if missing_packages:
            print(f"\n⚠ Packages manquants: {missing_packages}")
            print("Installation recommandée:")
            print("pip install pandas numpy scikit-learn opencv-python joblib")
        else:
            print("\n✓ Toutes les dépendances sont installées")

        return True

    except Exception as e:
        print(f"\n❌ Erreur lors du test: {e}")
        return False

def print_usage_instructions():
    """Affiche les instructions d'utilisation"""
    print("\n" + "=" * 60)
    print("INSTRUCTIONS D'UTILISATION")
    print("=" * 60)

    print("\n1. PREPROCESSING SEUL:")
    print("   python src/preprocessing/main_pipeline.py")

    print("\n2. ENTRAÎNEMENT COMPLET (preprocessing + modèle):")
    print("   python src/models/train_model.py")

    print("\n3. DEPUIS UN SCRIPT PYTHON:")
    print("""
   # Preprocessing seul
   from src.preprocessing.main_pipeline import DecisionTreePreprocessor
   preprocessor = DecisionTreePreprocessor()
   data = preprocessor.run_preprocessing()
   
   # Entraînement complet
   from src.models.train_model import DecisionTreeTrainer
   trainer = DecisionTreeTrainer()
   results = trainer.run_complete_training()
   """)

    print("\n4. MODIFICATION DE LA CONFIGURATION:")
    print("   Éditer le fichier config.json pour ajuster les paramètres")

    print("\n5. FICHIERS GÉNÉRÉS:")
    print("   - Logs: logs/YYMMDD-HHMMSS_*.log")
    print("   - Features: data/processed/features_for_dt.csv")
    print("   - Modèle: models/decision_tree_model.pkl")
    print("   - Évaluation: reports/decision_tree_evaluation.json")

def main():
    """Fonction principale"""
    success = test_configuration()

    if success:
        print_usage_instructions()
        print("\n" + "=" * 60)
        print("✅ CONFIGURATION PRÊTE POUR L'ENTRAÎNEMENT D'ARBRES DE DÉCISION")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ CONFIGURATION INCOMPLÈTE - VÉRIFIER LES ERREURS CI-DESSUS")
        print("=" * 60)

if __name__ == "__main__":
    main()
