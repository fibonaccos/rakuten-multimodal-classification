import streamlit as st

# Pour encapsuler le contenu dans des fonctions et des fichiers séparés,
# isolation des variables, etc si besoin
#from streamlit_components import conclusion


st.set_page_config(
    page_title="Conclusion",
    layout="wide",
    initial_sidebar_state="auto"
)


st.title("Conclusion")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Limites")
    st.write("##### Générales :")
    st.write("- Mise en évidence de la complexité de l'exploration et préparation des données réelles.")
    st.write("- Complication du développement dûe aux ressources computationnelles limités et à la collaboratioin à distance")
   
    st.write("##### Propres aux modèles :")
    st.write("- Random Forest : inadapté à la haute dimensionnalité textuelle")
    st.write("- SGDC : performance plafonnée (75% max) + ne capture pas les patterns non linéaires")
    st.write("- EfficientNet : ressources limitées + performances passables")

with col2:
    st.subheader("Ouvertures")
    st.write("- Amélioration nécessaire du prétraitement et de la création de variables")
    st.write("- Optimisation des choix de modèles et des hyper-paramètres à prévoir")
    st.write("- Multimodèles : vote à la majorité et/ou utilisation de réseaux de neuronnes denses")
    st.write("- Workflow et pratiques de collaboration à renforcer pour une meilleure efficacité")