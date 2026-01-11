import pandas as pd
import re
import unicodedata
import spacy
import html
import os
from sklearn.base import BaseEstimator, TransformerMixin

# --- CONFIGURATION ---
ACTIVER_TRADUCTION = False 

# On garde les imports légers (tqdm)
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, **kwargs): return iterator

# --- STOPWORDS (Liste définie manuellement) ---
STOPWORDS = set([
  "a", "à", "â", "abord", "absolument", "achat", "afin", "ah", "ai", "aie", "aient", "aies", "ailleurs", "ainsi", "ait", "allaient", "allo", "allons", "allô", "alors", "anterieur", "anterieure", "anterieures", "apres", "après", "article", "as", "assez", "attendu", "au", "aucun", "aucune", "aucuns", "aujourd", "aujourd'hui", "aupres", "auquel", "aura", "aurai", "auraient", "aurais", "aurait", "auras", "aurez", "auriez", "aurions", "aurons", "auront", "aussi", "autant", "autre", "autrefois", "autrement", "autres", "autrui", "aux", "auxquelles", "auxquels", "avaient", "avais", "avait", "avant", "avec", "avez", "aviez", "avions", "avoir", "avons", "ayant", "ayez", "ayons",
  "b", "bah", "bas", "basee", "bat", "beau", "beaucoup", "bien", "bigre", "blanc", "bleu", "bon", "boum", "boutique", "bravo", "brrr",
  "c", "ça", "car", "ce", "ceci", "cela", "celà", "celle", "celle-ci", "celle-là", "celles", "celles-ci", "celles-là", "celui", "celui-ci", "celui-là", "cent", "cependant", "certain", "certaine", "certaines", "certains", "certes", "ces", "cet", "cette", "ceux", "ceux-ci", "ceux-là", "chacun", "chacune", "chaque", "cher", "chers", "chez", "chiche", "chut", "chère", "chères", "ci", "cinq", "cinquantaine", "cinquante", "cinquantième", "cinquième", "clac", "clic", "cm", "coloris", "combien", "comme", "comment", "comparable", "comparables", "compris", "concernant", "contre", "couic", "couleur", "crac",
  "d", "da", "dans", "de", "debout", "dedans", "dehors", "deja", "delà", "depuis", "dernier", "derniere", "derriere", "derrière", "des", "description", "desormais", "desquelles", "desquels", "dessous", "dessus", "details", "deux", "deuxième", "deuxièmement", "devant", "devers", "devoir", "devra", "devrait", "different", "differentes", "differents", "différent", "différente", "différentes", "différents", "dimension", "dire", "directe", "directement", "disponible", "dit", "dite", "dits", "divers", "diverse", "diverses", "dix", "dix-huit", "dix-neuf", "dix-sept", "dixième", "doit", "doivent", "donc", "donner", "dont", "dos", "douze", "douzième", "dring", "droite", "du", "duquel", "durant", "dès", "début", "déjà", "désormais",
  "e", "effet", "egale", "egalement", "egales", "eh", "elle", "elle-même", "elles", "elles-mêmes", "en", "encore", "enfin", "entre", "envers", "environ", "es", "ès", "essai", "est", "et", "etant", "etc", "été", "étée", "étées", "étés", "etre", "être", "eu", "eue", "eues", "euh", "eur", "eurent", "euros", "eus", "eusse", "eussent", "eusses", "eussiez", "eussions", "eut", "eux", "eux-mêmes", "exactement", "excepté", "expedition", "extenso", "exterieur", "eûmes", "eût", "eûtes",
  "f", "faire", "fais", "faisaient", "faisant", "fait", "faites", "falloir", "faut", "façon", "feront", "fi", "flac", "floc", "fois", "font", "force", "furent", "fus", "fusse", "fussent", "fusses", "fussiez", "fussions", "fut", "fûmes", "fût", "fûtes",
  "g", "garanti", "garantie", "gens", "gratuit", "gratuite",
  "h", "ha", "haut", "hauteur", "hein", "hélas", "hem", "hep", "hi", "ho", "holà", "hop", "hormis", "hors", "hou", "houp", "hue", "hui", "huit", "huitième", "hum", "hurrah", "hé",
  "i", "ici", "il", "ils", "image", "importe", "inclus", "incluse",
  "j", "jaune", "je", "jusqu", "jusque", "juste",
  "k", "kg",
  "l", "la", "là", "laisser", "laquelle", "largeur", "las", "le", "lequel", "les", "lès", "lesquelles", "lesquels", "leur", "leurs", "livraison", "longtemps", "longueur", "lors", "lorsque", "lui", "lui-meme", "lui-même",
  "m", "ma", "magasin", "maint", "maintenant", "mais", "malgre", "malgré", "marque", "maximale", "me", "meme", "memes", "merci", "mes", "mien", "mienne", "miennes", "miens", "mille", "mince", "mine", "minimale", "ml", "mm", "modele", "moi", "moi-meme", "moi-même", "moindres", "moins", "mon", "mot", "moyennant", "multiple", "multiples", "même", "mêmes",
  "n", "na", "naturel", "naturelle", "naturelles", "ne", "neanmoins", "necessaire", "necessairement", "neuf", "neuvième", "ni", "noir", "nombreuses", "nombreux", "nommés", "non", "nos", "notamment", "notre", "nôtre", "nôtres", "nous", "nous-mêmes", "nouveau", "nouveaux", "nul", "néanmoins",
  "o", "ô", "occasion", "offre", "oh", "ohé", "olé", "ollé", "on", "ont", "onze", "onzième", "ore", "ou", "où", "ouf", "ouias", "oust", "ouste", "outre", "ouvert", "ouverte", "ouverts", "o|",
  "p", "paf", "pan", "par", "parce", "parfois", "parle", "parlent", "parler", "parmi", "parole", "parseme", "partant", "particulier", "particulière", "particulièrement", "pas", "passé", "pendant", "pense", "permet", "personne", "personnes", "peu", "peut", "peuvent", "peux", "pff", "pfft", "pfut", "photo", "pif", "pire", "pièce", "plein", "plouf", "plupart", "plus", "plusieurs", "plutôt", "poids", "possessif", "possessifs", "possible", "possibles", "pouah", "pour", "pourquoi", "pourrais", "pourrait", "pouvait", "pouvoir", "prealable", "precisement", "premier", "première", "premièrement", "prendre", "pres", "près", "prix", "probable", "probante", "procedant", "proche", "produit", "promo", "psitt", "pu", "puis", "puisque", "pur", "pure",
  "q", "qu", "quand", "quant", "quant-à-soi", "quanta", "quarante", "quatorze", "quatre", "quatre-vingt", "quatrième", "quatrièmement", "que", "quel", "quelconque", "quelle", "quelles", "quelqu'un", "quelque", "quelques", "quels", "qui", "quiconque", "quinze", "quoi", "quoique",
  "r", "rapide", "rare", "rarement", "rares", "ref", "reference", "relative", "relativement", "remarquable", "rend", "rendre", "restant", "reste", "restent", "restrictif", "retour", "revoici", "revoilà", "rien", "rouge",
  "s", "sa", "sacrebleu", "sait", "sans", "sapristi", "sauf", "savoir", "se", "sein", "seize", "selon", "semblable", "semblaient", "semble", "semblent", "sent", "sept", "septième", "sera", "serai", "seraient", "serais", "serait", "seras", "serez", "seriez", "serions", "serons", "seront", "ses", "seul", "seule", "seulement", "si", "sien", "sienne", "siennes", "siens", "sinon", "six", "sixième", "soi", "soi-même", "soient", "sois", "soit", "soixante", "sommes", "son", "sont", "sous", "souvent", "soyez", "soyons", "specifique", "specifiques", "speculatif", "stock", "stop", "strictement", "subtiles", "suffisant", "suffisante", "suffit", "suis", "suit", "suivant", "suivante", "suivantes", "suivants", "suivre", "sujet", "superpose", "sur", "surtout",
  "t", "ta", "tac", "taille", "tandis", "tant", "tardive", "te", "té", "tel", "telle", "tellement", "telles", "tels", "tenant", "tend", "tenir", "tente", "tes", "tic", "tien", "tienne", "tiennes", "tiens", "toc", "toi", "toi-même", "ton", "touchant", "toujours", "tous", "tout", "toute", "toutefois", "toutes", "treize", "trente", "tres", "très", "trois", "troisième", "troisièmement", "trop", "tsoin", "tsouin", "tu",
  "u", "un", "une", "unes", "uniformement", "unique", "uniques", "uns",
  "v", "va", "vais", "valeur", "vas", "vendeur", "venir", "vente", "vers", "vert", "via", "vif", "vifs", "vingt", "vivat", "vive", "vives", "vlan", "voici", "voie", "voient", "voilà", "voir", "voire", "vont", "vos", "votre", "vôtre", "vôtres", "vous", "vous-mêmes", "vu", "vé",
  "w",
  "x",
  "y",
  "z", "zut",
  "à", "â", "ça", "ès", "étaient", "étais", "était", "étant", "état", "étiez", "étions", "été", "étée", "étées", "étés", "êtes", "être", "ô"
])

# Chargement de Spacy
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    print("[Attention] Spacy non trouvé. Le nettoyage sera moins précis (Regex uniquement).")
    nlp = None

def remove_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

class NettoyeurTexte(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            data = X.copy()
            # Sécurité si les colonnes manquent
            if 'designation' not in data.columns: data['designation'] = ''
            if 'description' not in data.columns: data['description'] = ''
        else:
            data = pd.DataFrame({'description': X, 'designation': [''] * len(X)})

        texts_ready = []
        
        print("-------------------------------------------------")
        print(f"[NOUVEAU CODE] Démarrage du nettoyage rapide sur {len(data)} lignes.")
        print(f"Utilisation de {len(STOPWORDS)} mots vides (hardcodés).")
        print("-------------------------------------------------")
        
        # Boucle de nettoyage
        iterator = data.iterrows()
        if 'tqdm' in globals():
            iterator = tqdm(data.iterrows(), total=len(data), unit="lignes")

        for idx, row in iterator:
            desc = row['description'] if pd.notnull(row['description']) else ""
            desig = row['designation'] if pd.notnull(row['designation']) else ""
            
            # 1. Priorité Description, sinon Designation
            text = desc if str(desc).strip() else desig
            
            if not str(text).strip():
                texts_ready.append("")
                continue

            # 2. Nettoyage de base
            try:
                text = html.unescape(str(text))
            except:
                pass
                
            text = text.lower()
            text = remove_accents(text)

            # 3. Regex (Chiffres et lettres)
            text = re.sub(r'<[^>]+>', ' ', text)       # HTML
            text = re.sub(r'\b\w*\d\w*\b', ' ', text)  # Mots avec chiffres (ex: iphone12)
            text = re.sub(r'[^a-z\s]', ' ', text)      # Lettres a-z uniquement
            text = re.sub(r'\s+', ' ', text).strip()   # Espaces
            
            texts_ready.append(text)

        # 4. Lemmatisation (Spacy)
        if nlp:
            print("[Info] Lemmatisation en cours...")
            cleaned_texts = []
            # Batch processing
            batch_size = 1000
            for i in range(0, len(texts_ready), batch_size):
                batch = texts_ready[i:i+batch_size]
                for doc in nlp.pipe(batch, disable=["parser", "ner"]):
                    lemmas = [
                        t.lemma_ for t in doc 
                        if t.lemma_ not in STOPWORDS 
                        and len(t.lemma_) > 2 
                    ]
                    cleaned_texts.append(" ".join(lemmas))
                
                if (i // batch_size) % 10 == 0:
                    print(f"   > Traité {i}/{len(texts_ready)}...")
                    
            return cleaned_texts
        
        return texts_ready

def preprocess_data(input_path, output_path):
    print(f"[Preprocessing] Chargement : {input_path}")
    df = pd.read_csv(input_path)
    
    cleaner = NettoyeurTexte()
    df['description_propre'] = cleaner.transform(df)
    
    cols_to_save = ['description_propre']
    if 'prdtypecode' in df.columns: cols_to_save.append('prdtypecode')
    if 'productid' in df.columns: cols_to_save.append('productid')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[cols_to_save].to_csv(output_path, index=False)
    print(f"[Preprocessing] Sauvegardé avec succès : {output_path}")