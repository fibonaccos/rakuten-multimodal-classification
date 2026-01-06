import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from PIL import Image
import time
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont  # Pour dessiner sur l'image

# Fixer les graines aléatoires
seed = 42  # Vous pouvez choisir n'importe quel entier ici
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Ajustez l'affichage pour afficher toutes les colonnes
pd.set_option('display.max_columns', None)  # Affiche toutes les colonnes
pd.set_option('display.expand_frame_repr', False)  # Ne pas couper les lignes

# Constants
IMAGE_SIZE = (224, 224)  # Dimension des images
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ANCR_DIR = "/content/drive/MyDrive/Colab Notebooks/rakuten_projet/"
IMG_TEST_DIR = ANCR_DIR+'data/images/image_test/'

# Affichage des classes après encodage
encoded_classes = [18, 19, 16, 5, 8, 25, 11, 4, 23, 9, 13, 14, 22, 7, 21, 0, 24, 3, 6, 2, 15, 26, 10, 12, 1, 17, 20]
print("Classes encodées :", encoded_classes)

# Affichage des classes d'origine
original_classes = [2280, 2403, 2060, 1160, 1281, 2705, 1302, 1140, 2583, 1300, 1560, 1920, 
 2582, 1280, 2522, 10, 2585, 60, 1180, 50, 1940, 2905, 1301, 1320, 40, 
 2220, 2462]
print("Classes d'origine :", original_classes)

# Créer le mapping
label_mapping = dict(zip(encoded_classes, original_classes))

# Transformations des données
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Transformations des données
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom Dataset class
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Lister tous les fichiers d'image dans le répertoire
        self.image_filenames = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        # Retourner le nombre d'images
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Construction du nom de l'image
        img_name = os.path.join(self.root_dir, self.image_filenames[idx])

        # Chargement de l'image
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.image_filenames[idx]

# Charger EfficientNet
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True)  # Charger EfficientNet avec des poids pré-entraînés
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)  # Adapter la dernière couche

    def forward(self, x):
        return self.model(x)

def predict_classe(img_dir) :
  # Charger les datasets
  test_dataset = CustomImageDataset(img_dir,
                                    transform=transform_test)

  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

  # Validation du modèle
  model = EfficientNetModel(num_classes=27)
  model.to(DEVICE)
  try:
      model = torch.compile(model)
  except Exception:
      pass

  model.load_state_dict(torch.load(f'{ANCR_DIR}models/model_efficientNet.pth'))
  model.eval()  # Passer en mode évaluation
  with torch.no_grad():
    all_predictions = []
    all_probabilities = []

    for images, filename in test_loader:

        images = images.to(DEVICE)
        outputs = model(images)

        # Obtenir les probabilités avec softmax
        probabilities = torch.softmax(outputs, dim=1)

        # Obtenir les classes prédites
        _, predicted = torch.max(probabilities.data, 1)

        # Rassembler les prédictions et les probabilités
        all_predictions.extend(predicted.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().detach().numpy())

        idx = 0
        # Afficher les images avec les classes et probabilités
        for i in range(images.size(0)):
            # Récupérer l'image et la normaliser
            img_tensor = images[i].cpu().detach() # Obtention du tenseur d'image
            img_numpy = img_tensor.permute(1, 2, 0).numpy()  # Changer l'ordre des dimensions

            # Normaliser l'image (selon l'échelle de torchvision)
            img_numpy = (img_numpy - img_numpy.min()) / (img_numpy.max() - img_numpy.min())  # Normaliser entre 0 et 1

            # Convertir en entier 8 bits
            img_numpy = (img_numpy * 255).astype(np.uint8)

            # Ajouter le texte directement sur le tenseur ou calculer la position pour l'affichage
            predicted_class = all_predictions[-images.size(0) + i]
            # Mapper les valeurs
            mapped_class = label_mapping[predicted_class]
            probability = all_probabilities[-images.size(0) + i][predicted_class]
            text = f'Class: {mapped_class}, Prob: {probability:.2f}'

            # Créer une nouvelle figure pour sauvegarder
            plt.figure()
            plt.imshow(img_numpy)
            plt.title(text)  # Mettre le titre avec la classe et la probabilité
            plt.axis('off')  # Ne pas afficher les axes
            #plt.show()

            # Sauvegarder l'image
            os.makedirs(f"{IMG_TEST_DIR}image_predict", exist_ok=True)
            file_name = f'predicted_{filename[idx]}'  # Nommez le fichier comme bon vous semble
            print(f"{file_name}")
            plt.savefig(os.path.join(f"{IMG_TEST_DIR}image_predict", file_name), bbox_inches='tight', pad_inches=0)
            plt.close()  # Fermer la figure

            idx += 1


    print(f"Images prédites enregistrées")

predictions = predict_classe(IMG_TEST_DIR)