import os
import time
import re
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms, models

##### Préparation des images 

# Chemin vers le dossier contenant les images
dossier_images = "/Users/grizzly/Desktop/DataScientest/rakuten-multimodal-classification/data/images/image_train"

# Liste des fichiers dans le dossier
fichiers_images = os.listdir(dossier_images)

# Filtrer pour ne garder que les fichiers d'images (par exemple, .jpg)
fichiers_images = [f for f in fichiers_images if f.endswith('.jpg')]

# Prendre les 25 premières images
nb_img = len(fichiers_images)
#images_a_inferer = fichiers_images[:100]
images_a_inferer = fichiers_images[:45000]

##### Chargement du modèle ResNet

resnet101_model = models.resnet101(pretrained=True)
resnet101_model.eval()  # Mettre le modèle en mode évaluation

# Créer un pipeline de prétraitement
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

##### Inférence sur les images

resnet_pred_data = []

# Démarrer le timer
start_time = time.time()

for i, fichier in enumerate(images_a_inferer):
    # Récupérer l'imageid et le productid de l'image
    match = re.match(r'image_(\d+)_product_(\d+)\.jpg', fichier)
    if match:
        image_id = match.group(1)
        product_id = match.group(2)

        # Ouvrir l'image
        image = Image.open(os.path.join(dossier_images, fichier))

        img_preprocessed = preprocess(image)
        batch_img_tensor = torch.unsqueeze(img_preprocessed, 0)

        # Inférence
        out = resnet101_model(batch_img_tensor)

        # Charger les étiquettes
        with open("/Users/grizzly/Desktop/DataScientest/rakuten-multimodal-classification/data/imagenet_classes.txt") as f:
            labels = [line.strip() for line in f.readlines()]

        _, index = torch.max(out, 1)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

        # Obtenir les 5 meilleures prédictions
        _, indices = torch.sort(out, descending=True)
        top_predictions = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]

        # Afficher les résultats
        print(f"Image {i + 1} / 45000")
        #print(f"Image ID: {image_id}, Product ID: {product_id}, Prediction: {labels[index[0]]}, Confidence: {percentage[index[0]].item():.2f}%")

        # Ajouter les résultats à la liste
        resnet_pred_data.append({
            "imageid": image_id,
            "productid": product_id,
            "resnet_pred": labels[index[0]]
        })

# Créer un DataFrame à partir des résultats
resnet_pred_df = pd.DataFrame(resnet_pred_data)

# Sauvegarder le DataFrame dans un fichier CSV
resnet_pred_df.to_csv("data/resnet_pred.csv", index=False)

# Calculer et afficher le temps d'exécution
end_time = time.time()
execution_time = end_time - start_time
print(f"\nTemps d'exécution: {execution_time:.2f} secondes")
print(f"Nombre d'image : {nb_img}")
