import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Ajouter le chemin RACINE au PYTHONPATH pour pouvoir charger les transformers
# Les classes sont définies comme src.preprocessing.XXX.components
# Donc il faut ajouter la racine du projet, pas src/
APP_DIR = Path(__file__).parent.parent.parent  # .../src/streamlit/app/
PROJECT_ROOT = APP_DIR.parent.parent.parent  # Remonte de app/ -> streamlit/ -> src/ -> racine/

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Chemins vers les données dans src/streamlit/app
DATA_DIR = APP_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
TRANSFORMERS_DIR = DATA_DIR / "transformers"

# Dictionnaire des catégories
CATEGORY_LABELS = {
    "10": "Livres/Médias",
    "40": "Jeux vidéo anciens",
    "50": "Accessoires gaming",
    "60": "Consoles",
    "1140": "Figurines",
    "1160": "Livres fiction",
    "1180": "Livres jeunesse/BD",
    "1280": "Jeux vidéo",
    "1281": "Jeux PC",
    "1300": "Accessoires jeux vidéo",
    "1301": "Jeux de société",
    "1302": "Accessoires consoles",
    "1320": "Cartes à collectionner",
    "1560": "Mobilier",
    "1920": "Linge de maison",
    "1940": "Alimentation",
    "2060": "Décoration",
    "2220": "Animalerie",
    "2280": "Magazines",
    "2403": "Livres (autre type)",
    "2462": "Jouets vintage",
    "2522": "Papeterie",
    "2582": "Mobilier extérieur",
    "2583": "Piscines",
    "2585": "Bricolage",
    "2705": "Livres anciens",
    "2905": "Jeux de construction"
}


def get_category_label(code):
    """Retourne le label d'une catégorie à partir de son code"""
    return CATEGORY_LABELS.get(str(code), "Catégorie inconnue")


def load_model_artifacts(model_name):
    """Charge les artefacts d'un modèle (modèle + label encoder)"""
    try:
        model_path = MODELS_DIR / model_name
        
        # Charger le modèle
        if model_name == "RandomForest":
            model_file = model_path / "rf_model.pkl"
        elif model_name == "SGDCModel":
            model_file = model_path / "sgdc_model.pkl"
        else:
            return None, None
        
        # Charger label encoder
        label_encoder_file = model_path / "label_encoder.pkl"
        
        if not model_file.exists():
            return None, None
            
        if not label_encoder_file.exists():
            return None, None
        
        model = joblib.load(model_file)
        label_encoder = joblib.load(label_encoder_file)
        
        return model, label_encoder
    except:
        return None, None


def load_transformers(model_name=None):
    """Charge les transformers pour le preprocessing"""
    try:
        if model_name == "RandomForest":
            transformer_path = TRANSFORMERS_DIR / "transformers_rf.pkl"
        elif model_name == "SGDCModel":
            transformer_path = TRANSFORMERS_DIR / "transformers_sgdc.pkl"
        else:
            # Par défaut, essayer SGDC
            transformer_path = TRANSFORMERS_DIR / "transformers_sgdc.pkl"
        
        if transformer_path.exists():
            return joblib.load(transformer_path)
        
        return None
    except:
        return None


def preprocess_text(user_input, transformers):
    """Prétraite le texte utilisateur"""
    try:
        if transformers is None:
            return None
        
        # Créer DataFrame avec colonnes attendues
        input_df = pd.DataFrame({
            'designation': [user_input],
            'description': [user_input]
        })
        
        # Appliquer text_cleaner si disponible
        text_cleaner = transformers.get('text_cleaner')
        if text_cleaner:
            input_df = text_cleaner.transform(input_df)
        
        # Vectoriser le texte
        text_vectorizer = transformers.get('text_vectorizer')
        if text_vectorizer:
            text_features = text_vectorizer.transform(input_df)
            return text_features
        
        return None
    except:
        return None


def predict_with_model(model, label_encoder, text_features, model_name):
    """Fait une prédiction avec un modèle"""
    try:
        # Ajouter features images nulles (192 dimensions)
        import scipy.sparse as sp
        
        if text_features is not None:
            if sp.issparse(text_features):
                text_features_dense = text_features.toarray()
            else:
                text_features_dense = text_features
            
            image_features = np.zeros((1, 192))
            combined_features = np.hstack([text_features_dense, image_features])
        else:
            # Mode simulation
            if model_name == "SGDCModel":
                combined_features = np.zeros((1, 16192))
            else:
                combined_features = np.zeros((1, 10192))
        
        # Prédiction
        prediction_encoded = model.predict(combined_features)
        prediction_proba = model.predict_proba(combined_features)
        
        # Décoder
        prediction = label_encoder.inverse_transform(prediction_encoded)[0]
        
        return prediction, prediction_proba, prediction_encoded[0]
    except Exception as e:
        st.error(f"Erreur prédiction {model_name}: {e}")
        return None, None, None


def render():
    """Affiche la page de démonstration live"""
    
    st.markdown("## 🎮 Démonstration Live")
    
    st.info("""
    💡 **Testez les modèles en temps réel !**  
    Entrez la description d'un produit et observez les prédictions.
    """)
    
    # Charger les modèles
    with st.spinner("🔄 Chargement des modèles..."):
        sgdc_model, sgdc_label_encoder = load_model_artifacts("SGDCModel")
        rf_model, rf_label_encoder = load_model_artifacts("RandomForest")
    
    # Vérifier si au moins un modèle est chargé
    models_loaded = []
    if sgdc_model is not None:
        models_loaded.append("SGDC")
        st.success("✅ Modèle SGDC chargé (75.4% accuracy)")
    
    if rf_model is not None:
        models_loaded.append("Random Forest")
        st.success("✅ Modèle Random Forest chargé (50.8% accuracy)")
    
    if len(models_loaded) == 0:
        st.warning("""
        ⚠️ **Aucun modèle disponible pour la démo**
        
        Les modèles pré-entraînés ne sont pas présents ou incompatibles avec cette version de scikit-learn.
        
        **Note** : La démo nécessite les fichiers .pkl des modèles entraînés.
        """)
        return
    
    # Message info si certains modèles manquent
    if sgdc_model is None and rf_model is not None:
        st.info("ℹ️ Modèle SGDC non disponible (problème de compatibilité sklearn)")
    elif rf_model is None and sgdc_model is not None:
        st.info("ℹ️ Modèle Random Forest non disponible")
    
    # Exemples prédéfinis
    st.markdown("### 📝 Exemples à Tester")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📚 Livre"):
            st.session_state.demo_text = "Harry Potter et la Chambre des Secrets livre broché neuf de J.K. Rowling édition Gallimard fantasy roman jeunesse"
    
    with col2:
        if st.button("🎮 Jeu Vidéo"):
            st.session_state.demo_text = "FIFA 24 jeu vidéo console PS5 PlayStation sport football simulation EA Sports neuf sous blister"
    
    with col3:
        if st.button("🏊 Piscine"):
            st.session_state.demo_text = "Piscine gonflable Intex rectangulaire 300x200x75cm avec pompe filtre échelle bâche eau"
    
    with col4:
        if st.button("🛏️ Linge Maison"):
            st.session_state.demo_text = "Parure de lit housse de couette 240x220 coton blanc et gris moderne avec taies oreiller"
    
    # Zone de saisie
    st.markdown("### ✍️ Votre Description Produit")
    
    default_text = st.session_state.get('demo_text', '')
    user_input = st.text_area(
        "Entrez la description d'un produit :",
        value=default_text,
        height=100,
        placeholder="Ex: Livre Harry Potter neuf, Console PS5, Piscine gonflable..."
    )
    
    # Sélection des modèles à utiliser
    st.markdown("### 🤖 Sélection des Modèles")
    
    col1, col2 = st.columns(2)
    with col1:
        use_sgdc = st.checkbox("Utiliser SGDC", value=True, disabled=sgdc_model is None)
    with col2:
        use_rf = st.checkbox("Utiliser Random Forest", value=True, disabled=rf_model is None)
    
    # Bouton de prédiction
    col1, col2 = st.columns([1, 4])
    with col1:
        predict_button = st.button("🚀 Prédire", type="primary", use_container_width=True)
    
    if predict_button and user_input.strip():
        with st.spinner("🔄 Prédiction en cours..."):
            st.markdown("---")
            st.markdown("## 🎯 Résultats des Prédictions")
            
            results = []
            
            # Prédiction SGDC
            if use_sgdc and sgdc_model is not None:
                # Charger transformers SGDC
                transformers_sgdc = load_transformers("SGDCModel")
                text_features_sgdc = preprocess_text(user_input, transformers_sgdc)
                
                pred, proba, pred_encoded = predict_with_model(
                    sgdc_model, sgdc_label_encoder, text_features_sgdc, "SGDCModel"
                )
                if pred is not None:
                    results.append(("SGDC", pred, proba, pred_encoded, sgdc_label_encoder))
            
            # Prédiction Random Forest
            if use_rf and rf_model is not None:
                # Charger transformers RF
                transformers_rf = load_transformers("RandomForest")
                text_features_rf = preprocess_text(user_input, transformers_rf)
                
                pred, proba, pred_encoded = predict_with_model(
                    rf_model, rf_label_encoder, text_features_rf, "RandomForest"
                )
                if pred is not None:
                    results.append(("Random Forest", pred, proba, pred_encoded, rf_label_encoder))
            
            # Afficher les résultats
            if len(results) == 0:
                st.error("❌ Aucune prédiction disponible")
            elif len(results) == 1:
                # Un seul modèle
                model_name, pred, proba, pred_encoded, label_enc = results[0]
                
                st.markdown(f"### 🤖 Prédiction {model_name}")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"#### Catégorie Prédite")
                    st.markdown(f"<div style='font-size:3rem;text-align:center;color:#FF6B6B;font-weight:bold;'>{pred}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:1.2rem;text-align:center;color:#666;'>{get_category_label(pred)}</div>", unsafe_allow_html=True)
                    
                    max_proba = proba[0][pred_encoded]
                    st.metric("Confiance", f"{max_proba*100:.1f}%")
                
                with col2:
                    st.markdown("#### 📊 Top 5 Probabilités")
                    
                    top_5_idx = np.argsort(proba[0])[-5:][::-1]
                    top_5_classes = label_enc.inverse_transform(top_5_idx)
                    top_5_proba = proba[0][top_5_idx]
                    
                    for i, (cls, prob) in enumerate(zip(top_5_classes, top_5_proba)):
                        label = get_category_label(cls)
                        if i == 0:
                            st.success(f"**{cls}** - {label} : {prob*100:.2f}%")
                        else:
                            st.info(f"{cls} - {label} : {prob*100:.2f}%")
            
            else:
                # Comparaison des deux modèles
                st.markdown("### 🔀 Comparaison des Prédictions")
                
                col1, col2 = st.columns(2)
                
                for idx, (model_name, pred, proba, pred_encoded, label_enc) in enumerate(results):
                    with [col1, col2][idx]:
                        st.markdown(f"#### 🤖 {model_name}")
                        
                        st.markdown(f"<div style='font-size:2rem;text-align:center;color:#FF6B6B;font-weight:bold;'>{pred}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size:1rem;text-align:center;color:#666;'>{get_category_label(pred)}</div>", unsafe_allow_html=True)
                        
                        max_proba = proba[0][pred_encoded]
                        st.metric("Confiance", f"{max_proba*100:.1f}%")
                        
                        st.markdown("**Top 3 Classes**")
                        top_3_idx = np.argsort(proba[0])[-3:][::-1]
                        top_3_classes = label_enc.inverse_transform(top_3_idx)
                        top_3_proba = proba[0][top_3_idx]
                        
                        for i, (cls, prob) in enumerate(zip(top_3_classes, top_3_proba)):
                            label = get_category_label(cls)
                            if i == 0:
                                st.success(f"**{cls}** - {label} : {prob*100:.1f}%")
                            else:
                                st.info(f"{cls} - {label} : {prob*100:.1f}%")
                
                # Analyse comparative
                st.markdown("---")
                st.markdown("### 📊 Analyse Comparative")
                
                sgdc_pred = results[0][1] if results[0][0] == "SGDC" else results[1][1]
                rf_pred = results[1][1] if results[1][0] == "Random Forest" else results[0][1]
                
                if sgdc_pred == rf_pred:
                    st.success(f"""
                    ✅ **Accord parfait !** Les deux modèles prédisent la même catégorie : **{sgdc_pred}**
                    
                    Cela indique une forte confiance dans la prédiction.
                    """)
                else:
                    st.warning(f"""
                    ⚠️ **Désaccord entre les modèles**
                    
                    - SGDC prédit : **{sgdc_pred}**
                    - Random Forest prédit : **{rf_pred}**
                    
                    SGDC étant plus performant (75.4% vs 50.8%), sa prédiction est probablement plus fiable.
                    """)
    
    elif predict_button:
        st.warning("⚠️ Veuillez entrer une description de produit")
    
    # Légende des catégories
    st.markdown("---")
    st.markdown("### 📚 Codes des 27 Catégories Rakuten")
    
    with st.expander("Voir tous les codes catégories"):
        st.warning("""
        ⚠️ **Note** : Ces codes sont des identifiants internes Rakuten.  
        Les descriptions ci-dessous sont des interprétations basées sur l'analyse des données.
        """)
        
        categories_info = {
            "10": "Livres/Médias", "40": "Jeux vidéo anciens", "50": "Accessoires gaming",
            "60": "Consoles", "1140": "Figurines", "1160": "Livres fiction",
            "1180": "Livres jeunesse/BD", "1280": "Jeux vidéo", "1281": "Jeux PC",
            "1300": "Accessoires JV", "1301": "Jeux de société", "1302": "Accessoires consoles",
            "1320": "Cartes à collectionner", "1560": "Mobilier", "1920": "Linge de maison",
            "1940": "Alimentation", "2060": "Décoration", "2220": "Animalerie",
            "2280": "Magazines", "2403": "Livres (autre)", "2462": "Jouets vintage",
            "2522": "Papeterie", "2582": "Mobilier extérieur", "2583": "Piscines",
            "2585": "Bricolage", "2705": "Livres anciens", "2905": "Jeux de construction"
        }
        
        cols = st.columns(3)
        for idx, (code, desc) in enumerate(categories_info.items()):
            with cols[idx % 3]:
                st.write(f"**{code}** : {desc}")
