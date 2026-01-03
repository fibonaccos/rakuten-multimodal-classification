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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import time
import random
import matplotlib.pyplot as plt
import subprocess

subprocess.run(['python3', 'src/models/data_split_into_fold.py'], check=True)

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
ANCR_DIR = "/Users/grizzly/Desktop/DataScientest/rakuten_clean/"

# Custom Dataset class
class CustomImageDataset(Dataset):
    def __init__(self, X_dataframe, y_dataframe, root_dir, transform=None):
        self.X_dataframe = X_dataframe
        self.y_dataframe = y_dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.X_dataframe)

    def __getitem__(self, idx):
        # Construction du nom de l'image
        imageid = self.X_dataframe.iloc[idx, 1]  # index pour imageid
        productid = self.X_dataframe.iloc[idx, 0]  # index pour productid
        label = self.y_dataframe.iloc[idx]['prdtypecode']  # Utiliser le label encodé
        img_name = os.path.join(self.root_dir, f'image_{imageid}_product_{productid}.jpg')

        # Chargement de l'image
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)
            #print(f"random : {random.randint(0, 90)}")

        return image, label

# Transformations des données
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.rotate(random.choice([0, random.randint(0, 90)]))),  # Rotation fixe
    transforms.Lambda(lambda img: img.transpose(method=Image.FLIP_LEFT_RIGHT) if random.choice([True, False]) else img),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Charger les données et encoder les labels
X_train_df = pd.read_csv(f"{ANCR_DIR}data/train/X_train_val.csv")
X_val_df = pd.read_csv(f"{ANCR_DIR}data/test/X_test_val.csv")

y_train_df = pd.read_csv(f"{ANCR_DIR}data/train/y_train_val.csv")
y_val_df = pd.read_csv(f"{ANCR_DIR}data/test/y_test_val.csv")

X_val_df, X_test_df, y_val_df, y_test_df = train_test_split(X_val_df, y_val_df, stratify = y_val_df, test_size = 1/2, random_state = 42)

# Encoder les labels
label_encoder = LabelEncoder()
y_train_df['prdtypecode'] = label_encoder.fit_transform(y_train_df['prdtypecode'])
y_val_df['prdtypecode'] = label_encoder.transform(y_val_df['prdtypecode'])
y_test_df['prdtypecode'] = label_encoder.transform(y_test_df['prdtypecode'])

# Charger les datasets
train_dataset = CustomImageDataset(X_train_df, y_train_df,
                                    ANCR_DIR+'data/images/image_train',
                                    transform=transform_train)

val_dataset = CustomImageDataset(X_val_df, y_val_df,
                                ANCR_DIR+'data/images/image_train',
                                transform=transform_val)

test_dataset = CustomImageDataset(X_test_df, y_test_df,
                                ANCR_DIR+'data/images/image_train',
                                transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Charger EfficientNet
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True)  # Charger EfficientNet avec des poids pré-entraînés
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)  # Adapter la dernière couche

    def forward(self, x):
        return self.model(x)

# Validation du modèle
model = EfficientNetModel(num_classes=len(np.unique(y_train_df['prdtypecode'])))
model.to(DEVICE)
try:
    model = torch.compile(model)
except Exception:
    pass

model.load_state_dict(torch.load(f'{ANCR_DIR}models/EfficientNet/model_efficientNet.pth'))
model.eval()  # Passer en mode évaluation
all_labels = []
all_predictions = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # Prendre la classe avec la plus forte probabilité

        all_labels.extend(labels.cpu().numpy())       # Rassembler toutes les étiquettes
        all_predictions.extend(predicted.cpu().numpy())  # Rassembler toutes les prédictions

print(f"Precision : {precision_score(all_labels, all_predictions, average='weighted')}")
print(f"Recall : {recall_score(all_labels, all_predictions, average='weighted')}")
print(f"F1-score : {f1_score(all_labels, all_predictions, average='weighted')}")
print(f"Accuracy : {accuracy_score(all_labels, all_predictions)}")

# Affichage des classes d'origine
encoded_classes = y_train_df['prdtypecode'].unique()
print("Classes encodées :", encoded_classes)

# Affichage des classes après encodage
original_classes = label_encoder.classes_
print("Classes d'origine :", original_classes)

# Créer le mapping
label_mapping = dict(zip(encoded_classes, original_classes))

# Combinez les listes en une liste de tuples
combined = list(zip(encoded_classes, original_classes))

# Triez par la première liste
combined.sort()

# Séparez les listes triées
encoded_classes, original_classes = zip(*combined)

# Convertir en listes (si besoin)
encoded_classes = list(encoded_classes)
original_classes = list(original_classes)

print("Classes encodées :", encoded_classes)
print("Classes d'origine :", original_classes)

# Calcul de la matrice de confusion
cm = confusion_matrix(all_labels, all_predictions)

# Normalisation en pourcentage et conversion en entiers
cm_percentage_int = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100).round().astype(int)

# Utiliser le mapping pour créer le DataFrame de la matrice de confusion
df_cm = pd.DataFrame(cm_percentage_int,
                     index=original_classes,
                     columns=original_classes)

# Affichage avec une taille de figure plus grande
plt.figure(figsize=(10, 8))  # Ajustez (largeur, hauteur) selon vos besoins
disp = ConfusionMatrixDisplay(confusion_matrix=df_cm.values, display_labels=df_cm.index)
disp.plot(cmap='viridis', ax=plt.gca())

plt.title("Matrice de Confusion", size=16)
plt.xlabel("Classes Prédites", size=14)
plt.xticks(rotation=40, ha='right')  # Rotation à 45 degrés
plt.ylabel("Classes Réelles", size=14)

# Enregistrement de la matrice de confusion en tant qu'image
plt.savefig(f"{ANCR_DIR}/reports/EfficientNet/figures/matrice_de_confusion.png")
plt.show()
