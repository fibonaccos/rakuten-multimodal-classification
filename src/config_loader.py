import json
import os
from datetime import datetime
from typing import Dict, Any

def get_config(section: str = None) -> Dict[str, Any]:
    """
    Charge la configuration depuis le fichier config.json

    Args:
        section (str, optional): Section spécifique à charger (PREPROCESSING, LOGS, DECISION_TREE)
                                Si None, retourne toute la configuration

    Returns:
        Dict[str, Any]: Configuration demandée
    """
    # Chemin vers le fichier config.json
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Traitement spécial pour les logs avec {DATE}
        if section == "LOGS" and "{DATE}" in config.get("LOGS", {}).get("filePath", ""):
            date_str = datetime.now().strftime("%y%m%d-%H%M%S")
            config["LOGS"]["filePath"] = config["LOGS"]["filePath"].replace("{DATE}", date_str)

        if section:
            if section not in config:
                raise KeyError(f"Section '{section}' not found in config.json")
            return config[section]
        else:
            return config

    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")

def get_preprocessing_config() -> Dict[str, Any]:
    """Raccourci pour obtenir la configuration de preprocessing"""
    return get_config("PREPROCESSING")

def get_logs_config() -> Dict[str, Any]:
    """Raccourci pour obtenir la configuration des logs"""
    return get_config("LOGS")

def get_decision_tree_config() -> Dict[str, Any]:
    """Raccourci pour obtenir la configuration des arbres de décision"""
    return get_config("DECISION_TREE")

def validate_paths(config: Dict[str, Any]) -> bool:
    """
    Valide l'existence des chemins de données dans la configuration

    Args:
        config: Configuration de preprocessing

    Returns:
        bool: True si tous les chemins existent
    """
    base_path = os.path.dirname(os.path.dirname(__file__))
    paths = config.get("paths", {})

    missing_paths = []
    for path_name, path_value in paths.items():
        if path_name.startswith("raw"):  # Vérifier seulement les chemins d'entrée
            full_path = os.path.join(base_path, path_value)
            if not os.path.exists(full_path):
                missing_paths.append(f"{path_name}: {full_path}")

    if missing_paths:
        print(f"Chemins manquants: {missing_paths}")
        return False
    return True

def ensure_output_dirs(config: Dict[str, Any]) -> None:
    """
    Crée les répertoires de sortie s'ils n'existent pas

    Args:
        config: Configuration complète
    """
    base_path = os.path.dirname(os.path.dirname(__file__))

    # Créer les répertoires pour le preprocessing
    if "PREPROCESSING" in config:
        paths = config["PREPROCESSING"].get("paths", {})
        for path_name, path_value in paths.items():
            if path_name.startswith("clean"):
                full_path = os.path.join(base_path, path_value)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)

    # Créer les répertoires pour les logs
    if "LOGS" in config:
        log_path = config["LOGS"].get("filePath", "")
        if log_path:
            full_log_path = os.path.join(base_path, log_path)
            os.makedirs(os.path.dirname(full_log_path), exist_ok=True)

    # Créer les répertoires pour les modèles et sorties
    if "DECISION_TREE" in config:
        output_paths = config["DECISION_TREE"].get("output_paths", {})
        for path_value in output_paths.values():
            full_path = os.path.join(base_path, path_value)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
