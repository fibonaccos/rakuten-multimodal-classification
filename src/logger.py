import logging
import os
from typing import Any


def build_logger(name: str,
                 filepath: str,
                 baseformat: str,
                 dateformat: str,
                 level: Any = logging.INFO) -> logging.Logger:
    """
    Crée un logger configuré selon les paramètres fournis

    Args:
        name (str): Nom du logger
        filepath (str): Chemin vers le fichier log
        baseformat (str): Format des messages de log
        dateformat (str): Format de la date
        level: Niveau de logging

    Returns:
        logging.Logger: Logger configuré
    """
    # Créer le répertoire des logs s'il n'existe pas
    log_dir = os.path.dirname(filepath)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Créer le logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Éviter les doublons de handlers
    if logger.handlers:
        logger.handlers.clear()

    # Handler pour fichier
    file_handler = logging.FileHandler(filepath, encoding='utf-8')
    file_handler.setLevel(level)

    # Handler pour console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter(baseformat, datefmt=dateformat)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Ajouter les handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
