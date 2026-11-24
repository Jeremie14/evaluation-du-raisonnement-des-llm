from sentence_transformers import SentenceTransformer
import numpy as np


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

if __name__ == "__main__":

    modele = charger_modele()

    valeurs = [
        "Autonomie",
        "Liberté",
        "Responsabilité"
    ]

    principes = [
        "Participation collective",
        "Respect des individus"
    ]

    vecteur_valeurs = regrouper_embeddings(
        modele,
        valeurs
    )

    vecteur_principes = regrouper_embeddings(
        modele,
        principes
    )

    print("Dimensions des valeurs :", vecteur_valeurs.shape)
    print("Dimensions des principes :", vecteur_principes.shape)

    print("Vecteur des valeurs :", vecteur_valeurs[:5])
    print("Vecteur des principes :", vecteur_principes[:5])


