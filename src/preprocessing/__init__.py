"""
1. Copie des datasets
2. Renommage des colonnes sur tous les datasets
3. Traitement des datasets textuels :
    -> suppresion des caractères indésirables (spéciaux + non textuels (html par ex)) sur les colonnes concernées
    -> suppression des mots vides sur les colonnes concernées
    -> vectorisation des colonnes textuelles non vides (text embedding)
    -> complétion de la colonne description à partir des embeddings de title
    -> re-sampling des classes
    -> ajout de nouvelles features (lesquelles ?)
    -> ajout de l'identifiant de l'image associé (pour la classification multi-modale)
4. Traitement des datasets images:
    -> transformation des canaux
    -> détection de formes pour réduction de la taille effective de l'image (rendre sparse l'image pour réduire les dimensions)
    -> application de filtres convolutionnels classiques aux images (contour, contraste, etc)

Toutes les étapes incluant l'utilisation d'un algorithme qui fit sur les données doit être fitté sur les données d'entraînement seulement.
Toutes les étapes sont sujettes à une sauvegarde en cas de bug dans les étapes suivantes.
Possibilité de multi-threading sur 3. et 4. ?
Une fonction (minimum) par '->' ?
Une pipeline pour 3. et une autre pour 4. ?
"""
