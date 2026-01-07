import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Chargement des datasets
df1 = pd.read_csv('/Users/grizzly/Desktop/DataScientest/rakuten_clean/data/text/X_train_update.csv')  # Remplacez par le chemin de votre premier dataset
df2 = pd.read_csv('/Users/grizzly/Desktop/DataScientest/rakuten_clean/data/text/Y_train.csv')  # Remplacez par le chemin de votre deuxième dataset
print("ok1")

# Ajout d'une colonne pour les index
df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)

# Joindre les deux datasets sur l'index
df_merged = pd.concat([df1, df2], axis=1)

X = df_merged[["productid", "imageid"]][:12000]
y = df_merged["prdtypecode"][:12000]
print("ok2")


X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X, y, stratify = y, test_size = 1/6, random_state = 42)
X_train_fold1, X_test_fold1, y_train_fold1, y_test_fold1 = train_test_split(X_train_val, y_train_val, stratify = y_train_val, test_size = 1/5, random_state = 42)
X_temp, X_test_fold2, y_temp, y_test_fold2 = train_test_split(X_train_fold1, y_train_fold1, stratify = y_train_fold1, test_size = 1/4, random_state = 42)
X_temp, X_test_fold3, y_temp, y_test_fold3 = train_test_split(X_temp, y_temp, stratify = y_temp, test_size = 1/3, random_state = 42)
X_test_fold5, X_test_fold4, y_test_fold5, y_test_fold4 = train_test_split(X_temp, y_temp, stratify = y_temp, test_size = 1/2, random_state = 42)

X_train_fold2 = pd.concat([X_test_fold1, X_test_fold3, X_test_fold4, X_test_fold5], axis=0)
X_train_fold3 = pd.concat([X_test_fold1, X_test_fold2, X_test_fold4, X_test_fold5], axis=0)
X_train_fold4 = pd.concat([X_test_fold1, X_test_fold2, X_test_fold3, X_test_fold5], axis=0)
X_train_fold5 = pd.concat([X_test_fold1, X_test_fold2, X_test_fold3, X_test_fold4], axis=0)

y_train_fold2 = pd.concat([y_test_fold1, y_test_fold3, y_test_fold4, y_test_fold5], axis=0)
y_train_fold3 = pd.concat([y_test_fold1, y_test_fold2, y_test_fold4, y_test_fold5], axis=0)
y_train_fold4 = pd.concat([y_test_fold1, y_test_fold2, y_test_fold3, y_test_fold5], axis=0)
y_train_fold5 = pd.concat([y_test_fold1, y_test_fold2, y_test_fold3, y_test_fold4], axis=0)

# Créer les dossiers si ils n'existent pas
data_dir = '/Users/grizzly/Desktop/DataScientest/rakuten_clean/data'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
#val_dir = os.path.join(data_dir, 'val')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
#os.makedirs(val_dir, exist_ok=True)

# Dictionnaires pour stocker X et y
X_train_folds = {
    1: X_train_fold1,
    2: X_train_fold2,
    3: X_train_fold3,
    4: X_train_fold4,
    5: X_train_fold5,
}

y_train_folds = {
    1: y_train_fold1,
    2: y_train_fold2,
    3: y_train_fold3,
    4: y_train_fold4,
    5: y_train_fold5,
}

X_test_folds = {
    1: X_test_fold1,
    2: X_test_fold2,
    3: X_test_fold3,
    4: X_test_fold4,
    5: X_test_fold5,
}

y_test_folds = {
    1: y_test_fold1,
    2: y_test_fold2,
    3: y_test_fold3,
    4: y_test_fold4,
    5: y_test_fold5,
}

X_train_val.to_csv(os.path.join(train_dir, f'X_train_val.csv'), index=False)
y_train_val.to_csv(os.path.join(train_dir, f'y_train_val.csv'), index=False)
X_test_val.to_csv(os.path.join(test_dir, f'X_test_val.csv'), index=False)
y_test_val.to_csv(os.path.join(test_dir, f'y_test_val.csv'), index=False)

# Sauvegarder X_train
for i in range(1, 6):
    X_train_folds[i].to_csv(os.path.join(train_dir, f'X_train_fold{i}.csv'), index=False)

# Sauvegarder y_train
for i in range(1, 6):
    y_train_folds[i].to_csv(os.path.join(train_dir, f'y_train_fold{i}.csv'), index=False)

# Sauvegarder X_test
for i in range(1, 6):
    X_test_folds[i].to_csv(os.path.join(test_dir, f'X_test_fold{i}.csv'), index=False)

# Sauvegarder y_test (suppose que vous avez des y_test correspondants)
for i in range(1, 6):
    y_test_folds[i].to_csv(os.path.join(test_dir, f'y_test_fold{i}.csv'), index=False)

