import os
import time
import re
import pandas as pd
from ultralytics import YOLO

##### Préparation des images 

# Chemin vers le dossier contenant les images
dossier_images = "/Users/grizzly/Desktop/DataScientest/rakuten-multimodal-classification/data/images/image_train"

# Liste des fichiers dans le dossier
fichiers_images = os.listdir(dossier_images)

# Filtrer pour ne garder que les fichiers d'images (par exemple, .jpg)
fichiers_images = [f for f in fichiers_images if f.endswith('.jpg')]

# Prendre les 25 premières images
nb_img = len(fichiers_images)
images_a_inferer = fichiers_images[:45000]
#images_a_inferer = fichiers_images

##### Chargement du modèle YOLO

yolo_model = YOLO("yolo11n.pt")

# Dictionnaire des classes (exemple, à adapter selon votre modèle)
class_names = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "TV",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush"
}

##### Inférence sur les images

yolo_pred_data = []

# Démarrer le timer
start_time = time.time()

for i, fichier in enumerate(images_a_inferer):
    # Récupérer l'imageid et le productid de l'image
    match = re.match(r'image_(\d+)_product_(\d+)\.jpg', fichier)
    if match:
        image_id = match.group(1)
        product_id = match.group(2)

        results = yolo_model(os.path.join(dossier_images, fichier))  # Predict on an image

        # Initialiser une liste pour stocker les prédictions
        predictions = []

        for result in results:
            # Obtenir les boîtes, les scores et les classes
            boxes = result.boxes  # Obtenir les boîtes englobantes
            scores = boxes.conf  # Obtenir les scores de confiance
            classes = boxes.cls  # Obtenir les classes détectées

            # Ajouter les résultats à la liste des prédictions
            for j in range(len(boxes)):
                class_name = class_names.get(int(classes[j]), "Unknown")  # Obtenir le nom de la classe
                predictions.append({
                    "class": class_name,  # Nom de la classe
                    "score": float(scores[j]),  # Convertir en float
                    "box": boxes.xyxy[j].tolist()  # Convertir en liste
                })

        # Afficher les résultats
        list_class = []
        for pred in predictions:
            list_class.append(pred['class'])
            #print(f"Classe: {pred['class']}, Score: {pred['score']:.2f}, Boîte: {pred['box']}")

        # Afficher les résultats
        print(f"Image {i + 1} / 45000")
        #print(list_class)

        # Ajouter les résultats à la liste
        yolo_pred_data.append({
            "imageid": image_id,
            "productid": product_id,
            "yolo_pred": list_class  # Stocker les prédictions
        })

# Créer un DataFrame à partir des résultats
yolo_pred_df = pd.DataFrame(yolo_pred_data)

# Sauvegarder le DataFrame dans un fichier CSV
yolo_pred_df.to_csv("data/yolo_pred.csv", index=False)

# Calculer et afficher le temps d'exécution
end_time = time.time()
execution_time = end_time - start_time
print(f"\nTemps d'exécution: {execution_time:.2f} secondes")
print(f"Nombre d'images : {nb_img}")
