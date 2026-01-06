import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random
import matplotlib.pyplot as plt

# Reproductibility
seed = 42  
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Constants
IMAGE_SIZE = (224, 224) 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directory
ANCR_DIR = "./"
IMG_TEST_DIR = ANCR_DIR+'data/images/image_test/'

# Mapping
encoded_classes = [18, 19, 16, 5, 8, 25, 11, 4, 23, 9, 13, 14, 22, 7, 21, 0, 24, 3, 6, 2, 15, 26, 10, 12, 1, 17, 20]
original_classes = [2280, 2403, 2060, 1160, 1281, 2705, 1302, 1140, 2583, 1300, 1560, 1920, 2582, 1280, 2522, 10, 2585, 60, 1180, 50, 1940, 2905, 1301, 1320, 40, 2220, 2462]
label_mapping = dict(zip(encoded_classes, original_classes))

# Data transformation
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

# Class EfficientNet
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)  # Charger EfficientNet avec des poids pré-entraînés
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)  # Adapter la dernière couche

    def forward(self, x):
        return self.model(x)

# Prediction with EfficientNet
def predict_efficientNet(img_dir) :
  # Charger les datasets
  test_dataset = CustomImageDataset(img_dir, transform=transform_test)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

  # Chargement du modèle
  model = EfficientNetModel(num_classes=27)
  model.to(DEVICE)
  try:
      model = torch.compile(model)
  except Exception:
      pass

  model.load_state_dict(torch.load(f'{ANCR_DIR}models/EfficientNet/model_efficientNet.pth', map_location=DEVICE))
  
  # Prédiction du modèle
  model.eval()  # Passer en mode évaluation
  with torch.no_grad():
    all_predictions = []
    all_probabilities = []

    for images, filename in test_loader:
        # Récupération des images à tester
        images = images.to(DEVICE)
        outputs = model(images)

        # Obtenir les probabilités et les classes prédites
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities.data, 1)

        # Rassembler les prédictions et les probabilités
        all_predictions.extend(predicted.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().detach().numpy())

        idx = 0
        # Afficher les images avec les classes et probabilités
        for i in range(images.size(0)):
            # Récupérer l'image
            img_tensor = images[i].cpu().detach()
            img_numpy = img_tensor.permute(1, 2, 0).numpy()

            # Normaliser l'image
            img_numpy = (img_numpy - img_numpy.min()) / (img_numpy.max() - img_numpy.min())
            img_numpy = (img_numpy * 255).astype(np.uint8)

            # Récupération de la prédiction de l'image
            predicted_class = all_predictions[-images.size(0) + i]
            # Mapper les valeurs
            mapped_class = label_mapping[predicted_class]
            probability = all_probabilities[-images.size(0) + i][predicted_class]
            text = f'Class: {mapped_class}, Prob: {probability:.2f}'

            # Créer une nouvelle figure pour sauvegarder
            plt.figure()
            plt.imshow(img_numpy)
            plt.title(text)  # Mettre le titre avec la classe et la probabilité
            plt.axis('off') 

            # Sauvegarder l'image
            os.makedirs(f"{img_dir}image_predict", exist_ok=True)
            file_name = f'predicted_{filename[idx]}'
            plt.savefig(os.path.join(f"{img_dir}image_predict", file_name), bbox_inches='tight', pad_inches=0)
            plt.close()

            idx += 1

    print(f"Images prédites enregistrées dans le dossier : {img_dir}")

    return(True)

predictions = predict_efficientNet(IMG_TEST_DIR)

