from sentence_transformers import SentenceTransformer
import numpy as np

DIMENSIONS_THEMATIQUES = [
    "sujets",
    "domaines",
    "concepts",
    "problematique_centrale"
]

DIMENSIONS_EDITORIALES = [
    "idees_fondamentales",
    "valeurs",
    "principes",
    "vision_individu_societe",
    "rapport_au_changement",
    "approche_connaissance",
    "approche_argumentative",
    "posture_intellectuelle",
    "objectif_editorial"
]


def charger_modele():
    modele = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
    return modele

def vectoriser_caracteristiques(modele, caracteristiques):
    embeddings = modele.encode(caracteristiques, normalize_embeddings=True)
    return embeddings

def regrouper_embeddings(modele, caracteristiques):

    # Gestion d'une catégorie vide
    if len(caracteristiques) == 0:
        dimension = modele.get_sentence_embedding_dimension()
        return np.zeros(dimension, dtype=np.float32)

    # Vectorisation des caractéristiques
    embeddings = vectoriser_caracteristiques(
        modele,
        caracteristiques
    )

    # Mean Pooling
    vecteur_moyen = np.mean(embeddings, axis=0)
    return vecteur_moyen

def concatener_embeddings(vecteurs):
    return np.concatenate(vecteurs)

def extraire_caracteristiques(donnees):
    caracteristiques = []
    if isinstance(donnees, str):
        donnees = [donnees]
    for element in donnees:
        if isinstance(element, dict):
            texte = element.get("caracteristique", "")
        else:
            texte = str(element)
        if texte.strip():
            caracteristiques.append(texte)
    return caracteristiques

def vectoriser_article(modele, analyse):
    vecteurs = []

    # Caractéristiques thématiques
    for dimension in DIMENSIONS_THEMATIQUES:
        donnees = analyse.get("thematique", {}).get(dimension, [])
        caracteristiques = extraire_caracteristiques(donnees)
        vecteur = regrouper_embeddings(modele, caracteristiques)
        vecteurs.append(vecteur)

    # Caractéristiques éditoriales
    for dimension in DIMENSIONS_EDITORIALES:
        donnees = analyse.get("editorial", {}).get(dimension, [])
        caracteristiques = extraire_caracteristiques(donnees)
        vecteur = regrouper_embeddings(modele, caracteristiques)
        vecteurs.append(vecteur)

    # Concaténation des 13 vecteurs
    vecteur_article = concatener_embeddings(vecteurs)

    return vecteur_article


if __name__ == "__main__":

    modele = charger_modele()

    analyse_test = {
        "thematique": {
            "sujets": [
                {"caracteristique": "Éducation"}
            ],
            "domaines": [
                {"caracteristique": "Enseignement"}
            ],
            "concepts": [
                {"caracteristique": "Apprentissage"}
            ],
            "problematique_centrale":
                "Comment développer l'autonomie des étudiants?"
        },
        "editorial": {
            "valeurs": [
                {"caracteristique": "Autonomie"},
                {"caracteristique": "Liberté"}
            ],
            "principes": [
                {"caracteristique": "Respect des individus"}
            ]
        }
    }

    vecteur_article = vectoriser_article(
        modele,
        analyse_test
    )

    print("Dimensions de l'article :", vecteur_article.shape)
    print("Premières valeurs :", vecteur_article[:5])

    assert vecteur_article.shape == (4992,)

    print("Vectorisation réussie")


