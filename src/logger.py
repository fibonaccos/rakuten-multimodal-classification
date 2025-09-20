import logging
import os
from typing import Any


def build_logger(name: str,
                 filepath: str,
                 baseformat: str,
                 dateformat: str,
                 level: Any) -> logging.Logger:
    lgr = logging.getLogger(name)
    lgr.setLevel(level)
    if not lgr.handlers:
        # Construction correcte du chemin de fichier
        # Si filepath se termine par un underscore, on ajoute le nom
        # Sinon, on utilise filepath tel quel
        if filepath.endswith('_'):
            full_path = filepath + name + ".log"
        else:
            # Si filepath est un dossier, ajouter le nom du fichier
            if os.path.isdir(filepath) or filepath.endswith('/') or filepath.endswith('\\'):
                full_path = os.path.join(filepath, f"{name}.log")
            else:
                # filepath est déjà un chemin complet
                full_path = filepath

        # Créer le dossier si nécessaire
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        file_handler = logging.FileHandler(full_path, mode='w')
        formatter = logging.Formatter(baseformat, dateformat)
        file_handler.setFormatter(formatter)
        lgr.addHandler(file_handler)
    return lgr
