# Prétraitement des données pour le modèle DecisionTreeClassifier

Le modèle DecisionTreeClassifier utilise un arbre de décision pour la classification multiclasse. Le prétraitement combine des features textuelles (TF-IDF) et des features d'images (histogrammes de couleurs).

L'intégralité de la configuration du prétraitement est disponible et modifiable [ici](../../src/preprocessing/DecisionTreeModel/preprocessing.yaml).

## Configuration

La configuration est exposée sous format **yaml**. On y trouvera les éléments suivants :

- **metadata** : nom et description de la tâche associée.
- **preprocessing** : le cœur de la configuration.
  - **config** : paramètres généraux du prétraitement.
  - **input** : fichiers et dossiers utilisés par le prétraitement.
  - **output** : fichiers et répertoires créés à l'issue du prétraitement.
  - **steps** : transformations et étapes de prétraitement.

### Chemins

Le prétraitement nécessite l'accès à différents fichiers et répertoires (pré-existants ou non) :

- **preprocessing.config.logs.file_path** : chemin d'accès au fichier de log, *créé si besoin*. Il est conseillé de conserver le préfixe `{DATE}_` pour le nom du fichier.
- **preprocessing.input.text_path** : chemin d'accès aux données textuelles brutes.
- **preprocessing.input.image_dir** : chemin d'accès au dossier contenant les images brutes.
- **preprocessing.input.labels_path** : chemin d'accès aux labels bruts.
- **preprocessing.output.output_dir** : chemin d'accès au dossier de sortie, *créé si besoin*.
- **preprocessing.output.train_features** : chemin d'accès aux features d'entraînement, *créé si besoin*.
- **preprocessing.output.test_features** : chemin d'accès aux features de test, *créé si besoin*.
- **preprocessing.output.train_labels** : chemin d'accès aux labels d'entraînement, *créé si besoin*.
- **preprocessing.output.test_labels** : chemin d'accès aux labels de test, *créé si besoin*.
- **preprocessing.output.transformers** : chemin d'accès au fichier contenant les transformateurs, *créé si besoin*.

### Étapes de preprocessing

Le preprocessing se compose de plusieurs étapes configurables :

#### 1. Nettoyage du texte (`text_cleaning`)

- **enable** : activer/désactiver cette étape
- **lowercase** : conversion en minuscules
- **remove_punctuation** : suppression de la ponctuation
- **remove_stopwords** : suppression des mots vides

#### 2. Vectorisation TF-IDF (`tfidf_vectorization`)

- **enable** : activer/désactiver cette étape
- **max_features** : nombre maximum de features TF-IDF (3000 par défaut pour réduire la complexité)
- **ngram_range** : plage de n-grammes à considérer (ex: [1, 2] pour unigrammes et bigrammes)
- **min_df** : fréquence minimale de document
- **max_df** : fréquence maximale de document

#### 3. Features d'images (`image_features`)

- **enable** : activer/désactiver cette étape
- **extract_color_histograms** : extraction d'histogrammes de couleurs
- **extract_hog_features** : extraction de features HOG (non activé par défaut)
- **resize_shape** : taille de redimensionnement des images

## Exécution du preprocessing

Pour lancer le preprocessing, utilisez la commande suivante depuis la racine du projet :

```bash
python -m src.preprocessing.DecisionTreeModel
```

Le preprocessing génèrera :
- Les features d'entraînement et de test combinées (texte + images)
- Les labels d'entraînement et de test
- Les transformateurs sauvegardés pour la reproductibilité

## Structure des features générées

Les features finales combinent :
- **Features textuelles** : Vectorisation TF-IDF des colonnes `designation` et `description`
- **Features d'images** : Histogrammes de couleurs RGB (192 features par défaut : 64 bins × 3 canaux)

Le nombre total de features dépend de la configuration (typiquement 3000+ features TF-IDF + 192 features d'images).

## Considérations pour les arbres de décision

Les arbres de décision ont tendance à sur-apprendre avec un grand nombre de features. C'est pourquoi :
- Le nombre de features TF-IDF est réduit (3000 au lieu de 5000)
- Il est recommandé d'ajuster `max_depth`, `min_samples_split` et `ccp_alpha` lors de l'entraînement pour éviter le surapprentissage
