import streamlit as st
import pandas as pd
import json
from pathlib import Path
from PIL import Image
from models.EfficientNet.predict import predict_efficientNet
import os

# Configuration
st.set_page_config(
    page_title="Rakuten Classification",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
.main-header {font-size:2.5rem;font-weight:bold;color:#FF6B6B;text-align:center;padding:1rem;}
.metric-box {background:#f0f2f6;border-radius:10px;padding:1rem;margin:0.5rem 0;}
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_metrics(model_name):
    """Charge les métriques d'un modèle"""
    try:
        path = Path(f"models/{model_name}/metrics/metrics_summary.json")
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Erreur: {e}")
    return None

@st.cache_data
def load_image(path):
    """Charge une image"""
    if Path(path).exists():
        return Image.open(path)
    return None

# Navigation sidebar
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio(
    "Aller à:",
    [
        "Introduction",
        "Preprocessing",
        "Modélisation",
        "Résultats",
        "Demo",
        "Ouverture",
        "Conclusion"
    ],
    format_func=lambda x: {
        "Introduction": "🏠 Introduction",
        "Preprocessing": "🔧 Preprocessing",
        "Modélisation": "🤖 Modélisation",
        "Résultats": "📊 Résultats",
        "Demo": "🎮 Démo Live",
        "Ouverture": "🔮 Ouverture",
        "Conclusion": "✅ Conclusion"
    }.get(x, x)
)

# PAGE 1: INTRODUCTION
if page == "Introduction":
    st.markdown('<div class="main-header">🛍️ Classification Multimodale Rakuten</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Métriques clés
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Classes", value="27", delta="Catégories produits")
    with col2:
        st.metric(label="Échantillons", value="84 916", delta="Total dataset")
    with col3:
        st.metric(label="Features Max", value="16 192", delta="Dimensions SGDC")
    
    st.markdown("### 🎯 Objectif du Projet")
    st.write("""
    Développer un système de classification automatique capable de catégoriser 
    les produits Rakuten en utilisant à la fois les descriptions textuelles et les images.
    """)
    
    st.markdown("### 🔍 Problématique")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Challenge**
        - 27 catégories différentes
        - Données multimodales (texte + images)
        - Vocabulaire riche et varié
        - Classes déséquilibrées (ratio 1:50)
        """)
    with col2:
        st.success("""
        **Notre Approche**
        - TF-IDF pour le texte (8K features)
        - Histogrammes RGB pour images (192 features)
        - Modèles ML classiques (SGDC, DecisionTree, RF)
        - Optimisation hyperparamètres
        """)
    
    st.markdown("### 📊 Dataset Rakuten")
    data_overview = {
        "Métrique": ["Total produits", "Train (80%)", "Test (20%)", "Classes", "Déséquilibre max"],
        "Valeur": ["84 916", "67 933", "16 983", "27", "~1:50"]
    }
    st.table(pd.DataFrame(data_overview))
    
    st.markdown("### 👥 Contexte")
    st.write("**Projet DataScientest** - Formation Data Scientist - Janvier 2025")

# PAGE 2: PREPROCESSING
elif page == "Preprocessing":
    st.markdown('<div class="main-header">🔧 Pipeline de Preprocessing</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 📝 Preprocessing Textuel")
    
    with st.expander("⚙️ Étapes détaillées", expanded=True):
        st.markdown("""
        **1. Nettoyage**
        - Conversion en minuscules
        - Suppression de la ponctuation
        - Suppression des stopwords français
        
        **2. Vectorisation TF-IDF**
        - Unigrammes + bigrammes (1-2 mots)
        - max_features: 8000 (SGDC) / 5000 (DecisionTree)
        - min_df: 2 (minimum 2 documents)
        - max_df: 0.95 (maximum 95% documents)
        
        **3. Normalisation**
        - Normalisation L2 des vecteurs TF-IDF
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📄 Avant nettoyage**")
        st.code("""
Produit: Livre Harry Potter - État NEUF!!!
Prix: 15,99€ (Livraison GRATUITE)
Description: Roman fantasy pour jeunes...
        """, language="text")
    
    with col2:
        st.markdown("**✨ Après nettoyage**")
        st.code("""
livre harry potter état neuf
prix livraison gratuite  
roman fantasy jeunes
        """, language="text")
    
    st.markdown("### 🖼️ Preprocessing Images")
    
    with st.expander("⚙️ Extraction features visuelles", expanded=True):
        st.markdown("""
        **1. Redimensionnement**
        - Resize uniforme: 128×128 pixels
        
        **2. Histogrammes couleur RGB**
        - 64 bins par canal (R, G, B)
        - Total: 192 features numériques
        
        **3. Normalisation**
        - Min-Max scaling sur [0, 1]
        """)
    
    st.markdown("### 🔗 Features Finales par Modèle")
    
    features_comparison = {
        "Modèle": ["SGDC", "DecisionTree", "Random Forest"],
        "Features Texte": ["16 000", "10 000", "10 000"],
        "Features Image": ["192", "192", "192"],
        "Total Features": ["16 192", "10 192", "10 192"],
        "Échantillons": ["10 000", "5 000", "5 000"]
    }
    st.dataframe(pd.DataFrame(features_comparison), width='stretch')
    
    st.info("💡 **Note**: Le texte représente >98% des features. C'est normal pour un site e-commerce où les descriptions sont très informatives.")

# PAGE 3: MODÉLISATION  
elif page == "Modélisation":
    st.markdown('<div class="main-header">🤖 Modélisation</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 SGDClassifier", "🌳 DecisionTree", "🌲 Random Forest", "🧠 Deep Learning", "EfficientNet"])
    
    # TAB 1: SGDC
    with tab1:
        st.markdown("## 📈 SGDClassifier")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Principe de fonctionnement")
            st.write("""
            Modèle **linéaire** qui apprend progressivement en analysant les exemples  
            un par un (ou par mini-batches). Il trace 27 hyperplans dans un espace  
            à 16 192 dimensions pour séparer chaque catégorie de produit.
            """)
            
            st.markdown("#### Hyperparamètres")
            with st.expander("Voir la configuration", expanded=True):
                st.code("""
loss='log_loss'          # Fonction de perte logistique
penalty='elasticnet'     # Régularisation L1 + L2
alpha=0.00005           # Faible pénalisation
l1_ratio=0.15           # Mix 15% L1, 85% L2
max_iter=150            # 150 passages sur données
learning_rate='optimal' # Taux adaptatif
early_stopping=True     # Arrêt si stagnation
                """, language="python")
        
        with col2:
            metrics_sgdc = load_metrics("SGDCModel")
            if metrics_sgdc:
                st.metric("Accuracy", f"{metrics_sgdc.get('accuracy', 0)*100:.1f}%", delta="⭐ Excellent")
                st.metric("F1-Score", f"{metrics_sgdc.get('f1_weighted', 0)*100:.1f}%")
                st.metric("Precision", f"{metrics_sgdc.get('precision_weighted', 0)*100:.1f}%")
                st.metric("Recall", f"{metrics_sgdc.get('recall_weighted', 0)*100:.1f}%")
        
        st.markdown("#### Avantages & Inconvénients")
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **✅ Avantages**
            - Performance excellente (75.4%)
            - Scalable (millions d'exemples possibles)
            - Rapide (4 minutes training)
            - Pas de surapprentissage
            - Exploite haute dimensionnalité
            """)
        
        with col2:
            st.warning("""
            **⚠️ Inconvénients**
            - Interprétabilité limitée
            - Hypothèse de linéarité
            - Features images sous-exploitées
            - Nécessite preprocessing soigné
            """)
    
    # TAB 2: DecisionTree
    with tab2:
        st.markdown("## 🌳 DecisionTree")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Principe de fonctionnement")
            st.write("""
            Arbre de décisions qui pose des questions séquentielles sur les features.
            Comme un jeu de 20 questions pour deviner la catégorie du produit.
            """)
            
            st.markdown("#### Hyperparamètres")
            with st.expander("Voir la configuration", expanded=True):
                st.code("""
criterion='gini'         # Mesure d'impureté
max_depth=20            # Profondeur max 20 niveaux
min_samples_split=30    # Min 30 pour diviser
min_samples_leaf=15     # Min 15 par feuille
max_features=0.7        # 70% features par split
max_leaf_nodes=500      # Max 500 feuilles
ccp_alpha=0.001         # Pruning post-construction
                """, language="python")
        
        with col2:
            metrics_dt = load_metrics("DecisionTreeModel")
            if metrics_dt:
                st.metric("Accuracy", f"{metrics_dt.get('accuracy', 0)*100:.1f}%")
                st.metric("F1-Score", f"{metrics_dt.get('f1_weighted', 0)*100:.1f}%")
                st.metric("Overfitting Gap", f"{metrics_dt.get('overfitting_gap', 0)*100:.1f}%", delta="✅ Excellent")
        
        st.markdown("#### Avantages & Inconvénients")
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **✅ Avantages**
            - Interprétabilité maximale ⭐⭐⭐⭐⭐
            - Règles IF/THEN explicites
            - Ultra-rapide (5 secondes)
            - Pas d'overfitting (2.5% gap)
            - Visualisation possible
            """)
        
        with col2:
            st.warning("""
            **⚠️ Inconvénients**
            - Performance faible (40.9%)
            - Un seul arbre insuffisant
            - 5 classes jamais prédites
            - Haute dimensionnalité toxique
            """)
    
    # TAB 3: Random Forest
    with tab3:
        st.markdown("## 🌲 Random Forest")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Principe de fonctionnement")
            st.write("""
            Ensemble de 50 arbres de décision qui votent pour la prédiction finale.
            Améliore la performance en moyennant les décisions de multiples arbres.
            """)
            
            st.markdown("#### Hyperparamètres")
            with st.expander("Voir la configuration", expanded=True):
                st.code("""
n_estimators=50         # 50 arbres
max_depth=20            # Profondeur max
min_samples_split=30    
min_samples_leaf=15
max_features=0.7
                """, language="python")
        
        with col2:
            metrics_rf = load_metrics("RandomForest")
            if metrics_rf:
                st.metric("Accuracy", f"{metrics_rf.get('accuracy', 0)*100:.1f}%")
                st.metric("F1-Score", f"{metrics_rf.get('f1_weighted', 0)*100:.1f}%")
                st.metric("Overfitting Gap", f"{metrics_rf.get('overfitting_gap', 0)*100:.1f}%")
        
        st.markdown("#### Comparaison des 3 Modèles")
        comparison = {
            "Modèle": ["DecisionTree", "Random Forest", "SGDClassifier"],
            "Accuracy": ["40.9%", "50.8%", "75.4%"],
            "Temps Training": ["5 sec", "30 sec", "4 min"],
            "Interprétabilité": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐"],
            "Scalabilité": ["⭐", "⭐⭐", "⭐⭐⭐⭐⭐"]
        }
        st.dataframe(pd.DataFrame(comparison), width='stretch')
    
    # TAB 4: Deep Learning & Transfer Learning
    with tab4:
        st.markdown("## 🧠 Deep Learning & Transfer Learning")
        st.info("⚠️ Modèles en cours de développement sur branches séparées de l'équipe")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔬 TLModel (Transfer Learning)")
            st.markdown("**Branche** : `dev-fibonaccos-imagemodels`")
            
            st.markdown("#### Principe")
            st.write("""
            Utilisation d'architectures CNN pré-entraînées (ImageNet) avec fine-tuning  
            sur le dataset Rakuten. Apprentissage par transfert pour exploiter les  
            représentations visuelles apprises sur des millions d'images.
            """)
            
            st.metric("Accuracy Test", "12.8%", delta="⚠️ En cours d'optimisation", delta_color="off")
            st.metric("Weighted F1", "9.3%")
            
            st.markdown("#### État actuel")
            st.warning("""
            **Problèmes identifiés** :
            - Performance très faible (12.8% vs baseline 75%)
            - Possible sous-apprentissage ou problème de convergence
            - Dataset peut-être trop petit pour fine-tuning efficace
            - Hyperparamètres à optimiser
            
            **Travaux en cours** :
            - Augmentation de données (rotations, flips, crop)
            - Test de différentes architectures (ResNet, EfficientNet, ViT)
            - Ajustement learning rate et epochs
            """)
        
        with col2:
            st.markdown("### 🚀 EfficientNet")
            st.markdown("**Branche** : `rbanat_dev_efficientNet`")
            
            st.markdown("#### Principe")
            st.write("""
            Architecture CNN state-of-the-art qui scale efficacement profondeur,  
            largeur et résolution. Meilleur rapport performance/coût computationnel.
            """)
            
            st.metric("Statut", "En développement", delta="🔥 Prometteur")
            
            st.markdown("#### Caractéristiques")
            st.success("""
            **Avantages EfficientNet** :
            - Architecture optimale via Neural Architecture Search
            - Compound scaling (profondeur + largeur + résolution)
            - Performances SOTA sur ImageNet
            - Versions B0-B7 (scalabilité)
            
            **Application Rakuten** :
            - Embeddings visuels riches (2048D)
            - Reconnaissance patterns produits
            - Classification fine-grained
            """)
        
        st.markdown("### 📊 Comparaison Approches")
        
        comparison_dl = {
            "Approche": ["TF-IDF + ML", "TLModel (CNN)", "EfficientNet (projeté)", "Fusion Multimodale"],
            "Modalité": ["Texte", "Images", "Images", "Texte + Images"],
            "Accuracy": ["75.4%", "12.8%", "60-70% (estimé)", "80-85% (estimé)"],
            "Temps Training": ["4 min", "2-3h", "3-4h", "5-6h"],
            "Complexité": ["Faible", "Élevée", "Très élevée", "Très élevée"],
            "Statut": ["✅ Prod", "⚠️ Debug", "🔥 Dev", "🎯 Futur"]
        }
        st.dataframe(pd.DataFrame(comparison_dl), width='stretch')
        
        st.markdown("### 💡 Enseignements")
        
        st.markdown("""
        1. **Texte > Images** pour ce problème : Les descriptions Rakuten sont très informatives,  
           les images seules insuffisantes (packaging similaire entre catégories)
        
        2. **Deep Learning = Overhead** : Coût computationnel élevé (GPU requis) pour gain  
           modeste vs approche TF-IDF simple et efficace
        
        3. **Multimodalité prometteuse** : Combiner les deux modalités pourrait donner  
           80-85% mais nécessite architecture fusion complexe
        
        4. **ROI à considérer** : 
           - SGDC : 75.4% en 4 min (CPU) ✅
           - CNN : 60-70% en 4h (GPU) avec fine-tuning ⚠️
           - Fusion : 85% en 6h+ (GPU) ❓
        """)
        
        st.info("""
        💡 **Recommandation** : Pour un MVP production, SGDC reste le meilleur choix.  
        Le deep learning est pertinent si :
        - Budget GPU disponible
        - Besoin d'exploiter pleinement les images  
        - Viser les 5-10 derniers points d'accuracy
        """)



















    # TAB 5: EfficientNet
    with tab5:
        st.markdown("## 🧠 EfficientNet")

        tab51, tab52, tab53 = st.tabs(["🔍 Méthodologie", "Résultats", "Démonstration"])

        with tab51:
            st.markdown("## 🌟 Méthodologie")
            st.write("""
            Cette section présente la méthodologie adoptée pour notre étude, décrivant en détail les étapes clés du processus de prétraitement des données, 
            le choix des modèles, l'optimisation des hyperparamètres et les métriques de performance utilisées.
            """)

            tab511, tab512, tab513, tab514 = st.tabs(["⚙️ Pré-traitement des données", "📉 Choix des modèles", "🔧 Hyperparamètres", "📊 Métriques"])

            with tab511:
                st.markdown("### ⚙️ Pré-traitement des données")
                st.write("""
                Avant de commencer la phase d'entraînement, un **prétraitement** rigoureux des données est essentiel. Étant donné l'important déséquilibre des classes, 
                le **rééchantillonnage** a été une étape cruciale. Plusieurs méthodes de rééchantillonnage ont été envisagées :
                """)
                st.markdown("1. **Sous-échantillonnage des classes majoritaires** : Réduit le nombre d'échantillons dans les classes sur-représentées.")
                st.markdown("2. **Suréchantillonnage des classes minoritaires** : Augmente le nombre d'échantillons dans les classes sous-représentées.")
                st.markdown("3. **Approche hybride** : Combine les deux stratégies précédentes.")
                st.write("Nous avons testé à la fois le sous-échantillonnage et l’approche hybride, tout en laissant un pli intact pour la validation.")

            with tab512:
                st.markdown("### 📉 Choix des modèles")
                models_comparison = {
                    "Modèle": ["EfficientNet", "ResNet-50", "ResNet-101"],
                    "Description": [
                        "Architecture optimisée, excellente performance.",
                        "Facilite l’entraînement de réseaux très profonds.",
                        "Basé sur des blocs résiduels, améliore le passage des gradients."
                    ]
                }
                st.dataframe(pd.DataFrame(models_comparison))

            with tab513:
                st.markdown("### 🔧 Hyperparamètres")
                st.write("""
                Nous nous sommes concentrés sur le **tuning du learning rate** de chaque modèle testé dans une plage de **1e-2 à 1e-5**.
                Les méthodes de rééchantillonnage ont aussi été considérées comme des hyperparamètres.
                """)

                st.markdown("### Configuration du Learning Rate")
                with st.expander("Voir la configuration", expanded=True):
                    st.code("""
            learning_rate = 1e-3  # Exemple de learning rate défini
                    """, language="python")

            with tab514:
                st.markdown("### 📊 Métriques de comparaison")
                st.write("Le **F1-score pondéré** a été retenu comme métrique principale, avec d'autres métriques comme la précision, le rappel, et l'accuracy.")
                
                metrics_comparison = {
                    "Métrique": ["F1-score", "Précision", "Rappel", "Accuracy"],
                    "Description": [
                        "Intègre précision et rappel.",
                        "Pourcentage de vrais positifs parmi toutes les prédictions positives.",
                        "Capacité à identifier toutes les instances pertinentes.",
                        "Proportion globale de prédictions correctes."
                    ]
                }
                st.dataframe(pd.DataFrame(metrics_comparison))

        with tab52:
                st.markdown("## Résultats")
                st.write("""
                Cette section présente les performances des modèles testés, mettant en lumière l'impact des différentes approches de prétraitement, 
                des choix de modèles et de l'optimisation des hyperparamètres sur les métriques de performance.
                """)

                tab521, tab522, tab523, tab524, tab525 = st.tabs(["Impact du rééchantillonnage", "Sur-apprentissage", "Jeu de test", "Matrice de confusion", "Interprétabilité du modèle"])

                with tab521:
                    st.markdown("### Impact du rééchantillonnage")
                    st.image("./reports/EfficientNet/figures/crossval_report.png", caption="Comparaison des expériences")

                    st.write("""
                    Nous avons réalisé un total de \*\*5 expériences\*\* avec différentes approches de rééchantillonnage. Le tableau ci-dessous présente les résultats 
                    de ces méthodes au cours des validations croisées quant à l’entraînement décrit précédemment.
                    """)

                    st.write("""
                    Il est clair que le \*\*sous-échantillonnage\*\* est l'approche qui donne les \*\*meilleurs résultats\*\*, peu importe le \*\*learning rate\*\* utilisé. 
                    Pour les autres méthodes, l'approche utilisant des \*\*rotations et des transpositions\*\* s'avère légèrement supérieure à celle du \*\*doublement des données\*\* dans tous les cas.
                    """)

                    st.write("""
                    L’approche hybride obtient un \*\*F1-score\*\* de \*\*0,489\*\* en dupliquant les données contre \*\*0,494\*\* en appliquant des transformations. 
                    De même, en absence de rééchantillonnage, nous avons obtenu \*\*0,515\*\* avec duplication et \*\*0,528\*\* avec transformations.
                    """)

                    st.write("""
                    Il est intéressant de noter que ne pas rééchantillonner semble être la meilleure option dans notre cas, ce qui peut sembler contradictoire 
                    avec les bonnes pratiques de classification des données avec des classes déséquilibrées. Cette observation peut s'expliquer par le 
                    faible volume de données d'entraînement. En effet, avec la méthode de sous-échantillonnage, nous n'avons finalement que \*\*864 données\*\* 
                    sur les \*\*4 000 initiales\*\*, réduisant significativement le volume de données disponibles.
                    """)

                    st.write("""
                    En revanche, l’approche hybride a permis de conserver \*\*2 943 données\*\*, expliquant ainsi le meilleur score par rapport au sous-échantillonnage 
                    et les résultats légèrement inférieurs à ceux sans rééchantillonnage. On peut se demander si, avec un plus grand volume de données, 
                    l’approche hybride surpasserait celle sans rééchantillonnage.
                    """)

                with tab522:
                    st.markdown("### Sur-apprentissage")
                    st.image("./reports/EfficientNet/figures/learning_curve.png", caption="Courbe de sur-apprentissage")

                    st.write("""
                    Nous constatons un \*\*fort surapprentissage\*\*, ce qui était attendu au vu des résultats obtenus sur le jeu de validation. 
                    Avec l'utilisation de \*\*l'early stopping\*\*, le modèle final atteint un \*\*F1-score\*\* de \*\*0,590\*\* sur le jeu de validation 
                    après \*\*11 epochs\*\* et de \*\*0,901\*\* sur le jeu d’entraînement.
                    """)

                    st.write("""
                    Un point important à noter est que le \*\*F1-score\*\* sur le jeu de validation du modèle final a bien augmenté, passant de \*\*0,52\*\* à \*\*0,59\*\*. 
                    Cela suggère qu’avec plus de données, le surapprentissage pourrait se réduire de manière significative.
                    """)

                with tab523:
                    st.markdown("### Jeu de test")
                    st.write("Jeu de test : 1 000 images")
                    st.write("F1-score : 0,564")
                    st.write("Précision : 0,568")
                    st.write("Rappel : 0,569")
                    st.write("Accuracy : 0,569")

                    st.write("""
                    Les scores obtenus sur le jeu de test sont cohérents avec ceux de la validation précédente, sans mauvaises surprises.
                    """)

                with tab524:
                    st.image("./reports/EfficientNet/figures/confusion_matrix.png", caption="Matrice de confusion")

                    st.write("""
                    D’après la matrice de confusion, certaines classes (1160, 2522, et en particulier 60) sont très bien prédites par le modèle, 
                    avec un pourcentage de bonnes prédictions supérieur à **85%**, tandis que d’autres (1560, 1140, 1180, 2905) peinent à atteindre **30%**. 
                    Les autres classes affichent des taux variant entre **30%** et **76%**. En examinant de plus près la répartition des classes, il apparaît que les classes les mieux prédites sont sous-représentées dans notre échantillon.
                    En particulier, la classe 60 ne compte que 10 images sur les 1 000 du jeu de test. Cela pourrait indiquer des caractéristiques d’imagerie uniques, rendant ainsi cette classe « facilement » prévisible, même avec peu de données d’entraînement (1%).
                    À l'inverse, les classes 1180 et 2905 ne comprennent que 9 images chacune, suggérant qu'elles nécessitent davantage de données d’entraînement pour améliorer la capacité du modèle à les prédire correctement.
                    """)

                with tab525:
                    st.markdown("### Interprétabilité du modèle")
                    st.write("""
                    En raison de contraintes de temps et de ressources, la section dédiée à l'interprétabilité du modèle n'a pas pu être approfondie. 
                    Cependant, l’interprétabilité est cruciale pour comprendre les décisions prises par le modèle et évaluer sa confiance dans ses prédictions.
                    Dans de futurs travaux, il serait souhaitable d'explorer des méthodes telles que les **SHAP values** ou les **LIME** (Local Interpretable Model-agnostic Explanations) afin de mieux appréhender le fonctionnement du modèle.
                    """)

        with tab53:
            st.markdown("## Démonstration")
            # Ajoutez un bouton pour exécuter la fonction
            if st.button("Exécuter la prédiction"):
                result = predict_efficientNet('./data/images/image_test/')  # Appelle la fonction
                st.success(result)  # Affiche le résultat
                st.stop()

            # Bouton pour afficher les images
            if st.button("Afficher les images"):
                # Lister tous les fichiers d'image dans le répertoire
                image_files = [f for f in os.listdir('./data/images/image_test/image_predict') if f.endswith(('jpg', 'png', 'jpeg'))]

                # Vérifier s'il y a des images à afficher
                if len(image_files) == 0:
                    st.write("Aucune image trouvée dans le dossier.")
                else:
                    # Afficher chaque image
                    for image_file in image_files:
                        # Construire le chemin complet de l'image
                        image_path = os.path.join('./data/images/image_test/image_predict', image_file)

                        # Charger et afficher l'image
                        image = Image.open(image_path)
                        st.image(image, caption=image_file, width=500)  # Affiche l'image avec le nom comme légende























# PAGE 4: RÉSULTATS
elif page == "Résultats":
    st.markdown('<div class="main-header">📊 Résultats & Interprétation</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    model_choice = st.selectbox(
        "Sélectionner le modèle à analyser",
        ["SGDCModel", "DecisionTreeModel", "RandomForest"],
        index=0
    )
    
    metrics = load_metrics(model_choice)
    
    if metrics:
        st.markdown("### 📈 Métriques Globales")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
        with col2:
            st.metric("Precision", f"{metrics.get('precision_weighted', 0)*100:.1f}%")
        with col3:
            st.metric("Recall", f"{metrics.get('recall_weighted', 0)*100:.1f}%")
        with col4:
            st.metric("F1-Score", f"{metrics.get('f1_weighted', 0)*100:.1f}%")
        
        if 'train_accuracy' in metrics:
            st.markdown("### 📊 Analyse Overfitting")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Train Accuracy", f"{metrics.get('train_accuracy', 0)*100:.1f}%")
            with col2:
                st.metric("Test Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
            with col3:
                gap = metrics.get('overfitting_gap', 0) * 100
                delta_text = "✅ Excellent" if gap < 5 else ("⚠️ Attention" if gap < 15 else "❌ Problème")
                st.metric("Overfitting Gap", f"{gap:.1f}%", delta=delta_text)
        
        st.markdown("### 🎯 Matrice de Confusion")
        cm_path = Path(f"models/{model_choice}/metrics/confusion_matrix.png")
        if cm_path.exists():
            cm_img = load_image(cm_path)
            if cm_img:
                st.image(cm_img, use_column_width=True, caption=f"Matrice de confusion - {model_choice}")
        else:
            st.warning(f"⚠️ Matrice de confusion non disponible pour {model_choice}")
        
        st.markdown("### 🔍 Analyse Détaillée par Classe")
        
        report_path = Path(f"models/{model_choice}/metrics/classification_report.json")
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            
            # Extraire données des classes
            classes_data = []
            for cls, metrics_cls in report.items():
                if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                    classes_data.append({
                        "Classe": cls,
                        "Precision": f"{metrics_cls['precision']*100:.1f}%",
                        "Recall": f"{metrics_cls['recall']*100:.1f}%",
                        "F1-Score": f"{metrics_cls['f1-score']*100:.1f}%",
                        "Support": int(metrics_cls['support'])
                    })
            
            df_classes = pd.DataFrame(classes_data)
            df_classes['F1_numeric'] = df_classes['F1-Score'].str.rstrip('%').astype(float)
            df_classes = df_classes.sort_values('F1_numeric', ascending=False).drop('F1_numeric', axis=1)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🏆 Top 5 Classes (Meilleures)")
                st.dataframe(df_classes.head(5), width='stretch', hide_index=True)
                st.success("Ces classes ont un vocabulaire très distinctif")
            
            with col2:
                st.markdown("#### 😞 Bottom 5 Classes (Pires)")
                st.dataframe(df_classes.tail(5), width='stretch', hide_index=True)
                st.warning("Ces classes nécessitent plus de travail")
    else:
        st.error(f"❌ Métriques non trouvées pour {model_choice}")
    
    st.markdown("### 💡 Interprétabilité")
    
    if "SGDC" in model_choice:
        st.write("""
        **SGDClassifier** offre une interprétabilité **moyenne** via :
        - Coefficients des features (poids de chaque mot)
        - Feature importance globale
        - Analyse des mots les plus discriminants par classe
        
        ⚠️ **Limites** : 16K coefficients difficiles à interpréter individuellement
        """)
    else:
        st.write("""
        **Arbre de Décision** offre une interprétabilité **maximale** via :
        - Règles IF/THEN explicites et lisibles par humain
        - Visualisation graphique de l'arbre complet
        - Traçabilité complète de chaque décision
        - Export texte des règles (tree_structure.txt)
        
        ✅ **Avantage** : Chaque prédiction est 100% explicable
        """)

# PAGE 5: DÉMO LIVE
elif page == "Demo":
    st.markdown('<div class="main-header">🎮 Démo Live - Testez le Modèle</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.info("""
    💡 **Testez le modèle SGDC en temps réel !**  
    Entrez la description d'un produit et le modèle prédira sa catégorie.
    """)
    
    # Charger le modèle et les transformers
    @st.cache_resource
    def load_model_and_transformers():
        """Charge le modèle SGDC et les transformers"""
        try:
            import pickle
            
            # Charger le modèle
            with open('models/SGDCModel/artefacts/sgdc_model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            # Charger le label encoder
            with open('models/SGDCModel/artefacts/label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)
            
            # Charger les transformers (pour TF-IDF)
            with open('data/clean/sgdc_model/transformers.pkl', 'rb') as f:
                transformers = pickle.load(f)
            
            return model, label_encoder, transformers
        except Exception as e:
            st.error(f"Erreur lors du chargement : {e}")
            return None, None, None
    
    model, label_encoder, transformers = load_model_and_transformers()
    
    if model is not None:
        st.success("✅ Modèle SGDC chargé avec succès (75.4% accuracy)")
        
        # Exemples prédéfinis
        st.markdown("### 📝 Exemples à tester")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📚 Exemple Livre"):
                st.session_state.demo_text = "Harry Potter et la Chambre des Secrets livre broché neuf de J.K. Rowling édition Gallimard fantasy roman jeunesse"
        
        with col2:
            if st.button("🎮 Exemple Jeu Vidéo"):
                st.session_state.demo_text = "FIFA 24 jeu vidéo console PS5 PlayStation sport football simulation EA Sports neuf sous blister"
        
        with col3:
            if st.button("📱 Exemple High-Tech"):
                st.session_state.demo_text = "iPhone 15 Pro smartphone Apple 256GB noir titanium téléphone mobile 5G caméra 48MP écran OLED"
        
        # Zone de texte
        st.markdown("### ✍️ Votre description produit")
        
        default_text = st.session_state.get('demo_text', '')
        user_input = st.text_area(
            "Entrez la description d'un produit :",
            value=default_text,
            height=100,
            placeholder="Ex: Livre Harry Potter neuf, Console PS5, Smartphone Samsung Galaxy..."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            predict_button = st.button("🚀 Prédire", type="primary", use_container_width=True)
        with col2:
            st.write("")  # Spacer
        
        if predict_button and user_input.strip():
            with st.spinner("🔄 Prédiction en cours..."):
                try:
                    # Preprocessing du texte - CRÉER UN DATAFRAME
                    import pandas as pd
                    text_vectorizer = transformers['text_vectorizer']
                    
                    # Créer DataFrame avec les bonnes colonnes (designation et description)
                    input_df = pd.DataFrame({
                        'designation': [user_input],
                        'description': [user_input]  # On utilise le même texte pour les deux
                    })
                    
                    text_features = text_vectorizer.transform(input_df)
                    
                    # Comme on n'a pas d'image, on ajoute des features images nulles (192 dimensions)
                    import numpy as np
                    import scipy.sparse as sp
                    image_features = np.zeros((1, 192))
                    
                    # Combiner texte + image
                    if sp.issparse(text_features):
                        text_features_dense = text_features.toarray()
                    else:
                        text_features_dense = text_features
                    
                    combined_features = np.hstack([text_features_dense, image_features])
                    
                    # Prédiction
                    prediction_encoded = model.predict(combined_features)
                    prediction_proba = model.predict_proba(combined_features)
                    
                    # Décoder la prédiction
                    prediction = label_encoder.inverse_transform(prediction_encoded)[0]
                    
                    # Afficher les résultats
                    st.markdown("---")
                    st.markdown("## 🎯 Résultat de la Prédiction")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"### Catégorie Prédite")
                        st.markdown(f"<div style='font-size:3rem;text-align:center;color:#FF6B6B;font-weight:bold;'>{prediction}</div>", unsafe_allow_html=True)
                        
                        max_proba = prediction_proba[0][prediction_encoded[0]]
                        st.metric("Confiance", f"{max_proba*100:.1f}%")
                    
                    with col2:
                        st.markdown("### 📊 Top 5 Probabilités")
                        
                        # Obtenir les top 5 prédictions
                        top_5_idx = np.argsort(prediction_proba[0])[-5:][::-1]
                        top_5_classes = label_encoder.inverse_transform(top_5_idx)
                        top_5_proba = prediction_proba[0][top_5_idx]
                        
                        for i, (cls, prob) in enumerate(zip(top_5_classes, top_5_proba)):
                            if i == 0:
                                st.success(f"**{cls}** : {prob*100:.2f}%")
                            else:
                                st.info(f"{cls} : {prob*100:.2f}%")
                    
                    # Explication
                    st.markdown("---")
                    st.markdown("### 💡 Comment ça marche ?")
                    
                    st.write("""
                    1. **Preprocessing** : Votre texte est nettoyé et converti en vecteur TF-IDF (16K dimensions)
                    2. **Features images** : Simulées à zéro car pas d'image fournie (192 dimensions)
                    3. **Prédiction** : Le modèle SGDC calcule un score pour chacune des 27 classes
                    4. **Résultat** : La classe avec le score le plus élevé est retournée
                    """)
                    
                    # Afficher les mots-clés détectés
                    if hasattr(text_vectorizer, 'get_feature_names_out'):
                        feature_names = text_vectorizer.get_feature_names_out()
                        text_vector = text_features.toarray()[0]
                        
                        # Trouver les mots avec TF-IDF non nul
                        non_zero_idx = np.where(text_vector > 0)[0]
                        if len(non_zero_idx) > 0:
                            keywords = [(feature_names[i], text_vector[i]) for i in non_zero_idx]
                            keywords.sort(key=lambda x: x[1], reverse=True)
                            
                            st.markdown("### 🔑 Mots-clés détectés (Top 10)")
                            keywords_text = ", ".join([f"**{word}** ({score:.3f})" for word, score in keywords[:10]])
                            st.markdown(keywords_text)
                
                except Exception as e:
                    st.error(f"❌ Erreur lors de la prédiction : {e}")
                    st.exception(e)
        
        elif predict_button:
            st.warning("⚠️ Veuillez entrer une description de produit")
        
        # Légende des catégories
        st.markdown("---")
        st.markdown("### 📚 Codes des 27 Catégories Rakuten")
        
        with st.expander("Voir tous les codes (signification exacte non documentée)"):
            st.warning("""
            ⚠️ **Note** : Ces codes sont des identifiants internes Rakuten.  
            La signification précise de chaque code n'est pas fournie dans le dataset.  
            Les catégories ci-dessous sont des suppositions basées sur l'analyse des descriptions.
            """)
            
            categories_info = {
                "10": "Catégorie 10 - Livres/Médias",
                "40": "Catégorie 40 - Jeux vidéo anciens",
                "50": "Catégorie 50 - Accessoires gaming",
                "60": "Catégorie 60 - Consoles de jeux",
                "1140": "Catégorie 1140 - Figurines/Collectibles",
                "1160": "Catégorie 1160 - Livres (fiction/littérature)",
                "1180": "Catégorie 1180 - Livres jeunesse/BD",
                "1280": "Catégorie 1280 - Jeux vidéo",
                "1281": "Catégorie 1281 - Jeux PC",
                "1300": "Catégorie 1300 - Accessoires jeux vidéo",
                "1301": "Catégorie 1301 - Jeux de société",
                "1302": "Catégorie 1302 - Accessoires consoles",
                "1320": "Catégorie 1320 - Cartes à collectionner",
                "1560": "Catégorie 1560 - Mobilier",
                "1920": "Catégorie 1920 - Linge de maison",
                "1940": "Catégorie 1940 - Alimentation/Épicerie",
                "2060": "Catégorie 2060 - Décoration intérieure",
                "2220": "Catégorie 2220 - Animalerie",
                "2280": "Catégorie 2280 - Magazines/Presse",
                "2403": "Catégorie 2403 - Livres (autre type)",
                "2462": "Catégorie 2462 - Jeux et jouets vintage",
                "2522": "Catégorie 2522 - Papeterie/Fournitures",
                "2582": "Catégorie 2582 - Mobilier extérieur",
                "2583": "Catégorie 2583 - Piscines et accessoires",
                "2585": "Catégorie 2585 - Outillage/Bricolage",
                "2705": "Catégorie 2705 - Livres anciens/Collection",
                "2905": "Catégorie 2905 - Jeux de construction"
            }
            
            st.info("""
            💡 **Pourquoi plusieurs codes pour "Livres" ?**  
            Les codes 10, 1160, 2403, 2705 semblent tous liés aux livres mais correspondent  
            probablement à des sous-catégories différentes (genre, âge, format, etc.).  
            Sans la documentation Rakuten officielle, on ne peut que deviner la distinction exacte.
            """)
            
            cols = st.columns(3)
            for idx, (code, desc) in enumerate(categories_info.items()):
                with cols[idx % 3]:
                    st.write(f"**{code}** : {desc}")
    
    else:
        st.error("❌ Impossible de charger le modèle. Vérifiez que les fichiers sont présents.")
        st.info("""
        Fichiers requis :
        - `models/SGDCModel/artefacts/sgdc_model.pkl`
        - `models/SGDCModel/artefacts/label_encoder.pkl`
        - `data/clean/sgdc_model/transformers.pkl`
        """)

# PAGE 6: OUVERTURE
elif page == "Ouverture":
    st.markdown('<div class="main-header">🔮 Ouverture & Perspectives</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 🚀 Améliorations Possibles")
    
    st.info("""
    💡 **Note** : Plusieurs de ces améliorations sont **déjà en cours** par l'équipe sur d'autres branches !
    - Branche `dev-fibonaccos-imagemodels` : Transfer Learning (TLModel)
    - Branche `rbanat_dev_efficientNet` : EfficientNet
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🖼️ Améliorer les Features Images")
        st.info("""
        **Actuellement** : Histogrammes RGB (192 features)
        
        **Améliorations proposées** :
        - Embeddings ResNet50/EfficientNet (2048D)
        - Features HOG + SIFT combinées
        - Fine-tuning CNN sur dataset Rakuten
        - Vision Transformers (ViT)
        
        **Gain attendu** : +5 à 10 points d'accuracy
        """)
        
        st.markdown("#### 🤖 Ensemble Methods")
        st.info("""
        **Voting Classifier** combinant :
        - SGDC (75.4%)
        - Random Forest (50.8%)
        - XGBoost (60% estimé)
        
        **Vote pondéré ou majoritaire**
        
        **Gain attendu** : +3 à 5 points
        """)
    
    with col2:
        st.markdown("#### 📝 Améliorer le Texte")
        st.info("""
        **Actuellement** : TF-IDF (16K features)
        
        **Améliorations proposées** :
        - CamemBERT embeddings (768D contextuels)
        - FlauBERT fine-tuné sur Rakuten
        - Sentence-BERT français
        - GPT pour augmentation de données
        
        **Gain attendu** : +5 à 15 points d'accuracy
        """)
        
        st.markdown("#### ⚡ Optimisations Production")
        st.info("""
        **Pour le déploiement** :
        - ONNX runtime (3x plus rapide)
        - Quantization int8
        - Feature hashing (mémoire réduite)
        - API REST avec FastAPI
        
        **Résultat** : Production-ready avec scalabilité
        """)
    
    st.markdown("### 🎯 Roadmap Idéale (si plus de temps)")
    
    roadmap = {
        "Phase": ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5"],
        "Action": ["Baseline SGDC ✅", "Embeddings CNN", "CamemBERT texte", "Fusion multimodale", "Production API"],
        "Accuracy cible": ["75%", "80%", "85%", "88%", "90%+"],
        "Durée estimée": ["Fait", "1 semaine", "2 semaines", "1 semaine", "2 semaines"],
        "Priorité": ["✅", "🔥 Haute", "🔥 Haute", "📈 Moyenne", "🚀 Basse"]
    }
    st.table(pd.DataFrame(roadmap))
    
    st.markdown("### 🏆 Benchmark Industrie")
    
    benchmark_data = {
        "Approche": [
            "Random Guess (hasard)",
            "TF-IDF + ML classique (notre projet)",
            "CNN + RNN",
            "BERT + CNN multimodal",
            "État de l'art (ensemble deep learning)"
        ],
        "Accuracy": ["3.7%", "75.4%", "82%", "87%", "92%"],
        "Complexité": ["Nulle", "Faible", "Moyenne", "Élevée", "Très élevée"],
        "Temps training": ["-", "4 min", "2h", "6h", "12h+"],
        "Ressources": ["-", "CPU seul", "1 GPU", "Multi-GPU", "Cluster GPU"]
    }
    st.dataframe(pd.DataFrame(benchmark_data), width='stretch')
    
    st.success("""
    💡 **Conclusion** : Notre approche TF-IDF + SGDClassifier représente un excellent compromis  
    performance/complexité pour un MVP (Minimum Viable Product).
    
    **Gain vs hasard** : +71.7 points  
    **Écart vs deep learning** : -11 points seulement  
    **Rapport ressources** : 100x moins de ressources que deep learning
    """)

# PAGE 6: CONCLUSION
else:  # Conclusion
    st.markdown('<div class="main-header">✅ Conclusion</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 🎯 Objectifs Atteints")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("""
        **✅ Classification 27 classes**
        - 75.4% accuracy (SGDC)
        - 50.8% accuracy (Random Forest)
        - 40.9% accuracy (DecisionTree)
        """)
    with col2:
        st.success("""
        **✅ Approche Multimodale**
        - Texte via TF-IDF
        - Images via histogrammes RGB
        - Fusion features réussie
        """)
    with col3:
        st.success("""
        **✅ Modèles Robustes**
        - Pas d'overfitting
        - Scalables
        - Interprétables
        """)
    
    st.markdown("### 📊 Résultats Clés en Chiffres")
    
    results_summary = {
        "Métrique": ["Meilleur modèle", "Accuracy", "Gain vs hasard", "Temps training", "Interprétabilité"],
        "Valeur": ["SGDClassifier", "75.4%", "+71.7 points", "4 minutes", "Moyenne"],
        "Contexte": ["Modèle linéaire", "3 sur 4 corrects", "vs 3.7% random", "Sur CPU", "Via coefficients"]
    }
    st.table(pd.DataFrame(results_summary))
    
    st.markdown("### 💪 Points Forts du Projet")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Aspects Techniques**
        - Pipeline preprocessing robuste et réutilisable
        - Optimisation hyperparamètres systématique
        - Validation train/test rigoureuse
        - Documentation complète du code
        - Comparaison 3 approches différentes
        """)
    with col2:
        st.info("""
        **Aspects Métier**
        - Solution adaptée au contexte e-commerce
        - Scalable à des millions de produits
        - Temps de réponse acceptable (<1s)
        - Explicabilité pour audit/conformité
        - Coût infrastructure faible (CPU)
        """)
    
    st.markdown("### 🔄 Autocritique & Limites Identifiées")
    
    st.warning("""
    **Limites actuelles** :
    - Features images très basiques (histogrammes) → Potentiel CNN non exploité
    - Pas de deep learning (contrainte GPU/temps de formation)
    - 5 classes très difficiles (F1 < 20%) nécessitent attention particulière
    - Déséquilibre classes partiellement résolu seulement
    - Pas de validation croisée (k-fold) faute de temps
    """)
    
    st.markdown("### 🎓 Principaux Apprentissages")
    
    st.markdown("""
    1. **Le texte est roi en e-commerce** : Les descriptions sont plus discriminantes que les images pour ce problème
    2. **Simple peut largement suffire** : TF-IDF + SGDC rivalise avec des approches bien plus complexes
    3. **Régularisation est cruciale** : Éviter l'overfitting est plus important que performance brute
    4. **Importance des trade-offs** : Performance vs Interprétabilité, Complexité vs Temps, Coût vs Gain
    5. **Valeur d'une baseline solide** : Avoir un modèle ML simple avant de passer au deep learning
    """)
    
    st.markdown("### 🙏 Remerciements")
    st.write("""
    - **DataScientest** pour la formation et l'accompagnement
    - **Rakuten France** pour la mise à disposition du dataset
    - **L'équipe projet** pour la collaboration et l'entraide
    - **Les formateurs** pour leur expertise et leurs conseils
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 3rem 0;'>
        <h1>🎉 Merci de votre attention ! 🎉</h1>
        <h3>Des questions ?</h3>
    </div>
    """, unsafe_allow_html=True)

# FOOTER (toujours affiché)
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 1rem 0;'>
    <strong>Classification Multimodale Rakuten</strong> | DataScientest 2025 | Made with ❤️ and Streamlit
</div>
""", unsafe_allow_html=True)

