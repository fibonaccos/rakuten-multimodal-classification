
## MESSAGE DU COLLEGUE SUR DISCORD
`Salut la team, j'ai restructuré mon preprocessing pour le modèle de transfer learning, c'est bien plus propre et organisé. Pour l'instant c'est sur la branche dédié (dev-fibonaccos-imagemodels). Dans les grandes lignes :
sous-dossier dédié dans src/preprocessing pour le modèle (TLModel), qui est un module qu'on peut exécuter directement pour réaliser le preprocessing complet
le minimum de fichiers pour la simplicité et un fichier de configuration dédié au preprocessing uniquement
dossier docs/preprocessing à la racine du projet avec un markdown qui explique le fichier de configuration et comment lancer le preprocessing
`
`je pense que ça peut être pas mal si pour chaque modèle on a un truc du genre :
un / des fichier.s qui définit les transformations du preprocessing pour le modèle (ça irait dans le dossier src/preprocessing du github ?)
un / des fichier.s qui définit les nouvelles colonnes (features) construites si besoin (ça irait dans le dossier src/features du github ?)
un fichier qui exécute le.s pipeline.s de preprocessing pour le modèle (ça irait dans le dossier src/preprocessing du github ?)
un fichier qui définit le modèle (architecture) si besoin (ça irait dans le dossier src/models du github ?)
un fichier pour l'entraînement du modèle (en mode exécutable avec -m comme Steeve a fait) (ça irait dans le dossier src/models ou dans un sous-dossier src/models/train du github et le modèle entraîné dans le dossier models ?)
un fichier pour la prédiction du modèle (en mode exécutable avec -m comme Steeve a fait) (ça irait dans le dossier src/models ou dans un sous-dossier src/models/predict du github ?)
un / des fichiers pour la production de résultats annexes (interprétabilité, etc) (ça irait dans le dossier src/models ou dans un sous-dossier src/models/results ou dans le src/visualization du github ?)
un / des fichier.s pour les utilitaires (logs éventuels, etc) si besoin (ça irait dans un dossier src/utils du github ?)
un fichier de config (paramètres du preprocessing, du modèle, de l'entraînement, chemins des répertoires et fichiers, etc) pour la repro (ça irait dans un dossier src/config du github ?)
Ca permettrait de naviguer beaucoup plus facilement, et ça facilitera l'intégration des modèles qu'on veut dans le streamlit, vous en dites quoi ?`

## INSTRUCTIONS

Je veux que tu regardes comment c'est fait sur sa branche github. En gros on va avoir une présentation avec notre
groupe de travail dans quelques jours et il nous faut une très bonne organisation avec surtout,
des modèles fonctionnels et une interprétabilité au top. J'ai besoin que l'on réorganise comme le collègue
l'a fait sur sa branche (`dev-fibonaccos-imagemodels`); J'ai créé des copies de branches de sgdc_classifier (reorg_sgdc_classif)
et de arbre_decision (`reorg_arbre_decision`) qui sont des branches que j'ai moi même fait.

Je veux que tu ailles sur les branches dédiées à la réorganisation et que tu t'assures que quand
on va tous faire des pull requests pour merge sur la branche finale (qui n'existe pas encore), qu'on ait pas 
de merge conflict. Donc tu respectes la même organisation que le collègue sur sa branche.

En suite, tu t'assures que les modèles fonctionnent, que tout est OK (preprocessing, pipelines, architecture, interprétabilité au moyen de fichiers .md ou autre graphique avec librairies de ton choix)
En gros quand on va construire le streamlit il faut qu'on puisse lancer n'importe quel modèle et qu'on ait de la reproductibilité

N'hésite pas si t'as des questions

## IMPORTANT
- Ne toucher que les branches que j'ai mentionné de reorg
- La tâche sera terminée quand les deux branches seront réorganisées selon mes directives
- La tâche est terminée quand tous les modèles fonctionnnent
- La tâche est terminée quand on peut interpreter les résultats très facilement à l'aide de guide d'interprétabilité
- La tâche est terminée quand on peut lancer le preprocessing sans erreur
- La tâche est terminée quand les modèles fonctionnent et n'ont pas de surrapprentissage et que les résultats sont cohérents
- Tu peux modifier du code mais il faut conserver l'essence de base, le but n'est pas non plus de reprendre tout du début
- Tu peux aller chercher les fichiers des modèles sur les branches que tu veux (vérifie les fichiers partout)
