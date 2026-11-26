from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from vectorisation import(charger_modele, vectoriser_caracteristiques)
import matplotlib.pyplot as plt

def appliquer_kmeans(embeddings, nombre_clusters):
    modele_kmeans = KMeans(n_clusters=nombre_clusters, random_state=42, n_init=10)
    clusters = modele_kmeans.fit_predict(embeddings)
    return modele_kmeans, clusters

def methode_du_coude(embeddings, max_clusters=10):
    wcss = []
    max_clusters = min(max_clusters, len(embeddings))
    nombres_clusters = range(1, max_clusters + 1)
    for k in nombres_clusters:
        modele_kmeans, _ = appliquer_kmeans(embeddings, nombre_clusters=k)
        wcss.append(modele_kmeans.inertia_)
    return list(nombres_clusters), wcss

def calculer_silhouette(embeddings, max_clusters=10):

    scores = []

    max_clusters = min(
        max_clusters,
        len(embeddings) - 1
    )

    nombres_clusters = range(2, max_clusters + 1)

    for k in nombres_clusters:

        _, clusters = appliquer_kmeans(
            embeddings,
            nombre_clusters=k
        )

        score = silhouette_score(
            embeddings,
            clusters
        )

        scores.append(score)

    return list(nombres_clusters), scores

def reduire_dimensions(embeddings, nombre_dimensions=2):
    pca = PCA(n_components=nombre_dimensions)
    embeddings_reduit = pca.fit_transform(embeddings)
    return pca, embeddings_reduit

def afficher_methode_du_coude(nombres_clusters, inerties):

    plt.figure(figsize=(8, 5))

    plt.plot(
        nombres_clusters,
        inerties,
        marker="o"
    )

    plt.xlabel("Nombre de clusters (K)")
    plt.ylabel("Inertie")
    plt.title("Méthode du coude")
    plt.xticks(nombres_clusters)
    plt.grid(True)
    plt.tight_layout()

    plt.show()

def afficher_silhouette(nombres_clusters, scores):

    plt.figure(figsize=(8, 5))

    plt.plot(
        nombres_clusters,
        scores,
        marker="o"
    )

    plt.xlabel("Nombre de clusters (K)")
    plt.ylabel("Coefficient de silhouette")
    plt.title("Analyse du coefficient de silhouette")
    plt.xticks(nombres_clusters)
    plt.grid(True)
    plt.tight_layout()

    plt.show()

def afficher_clusters(embeddings_reduits, clusters, caracteristiques):

    plt.figure(figsize=(10, 7))

    plt.scatter(
        embeddings_reduits[:, 0],
        embeddings_reduits[:, 1],
        c=clusters,
        cmap="tab10",
        s=100
    )

    for i, caracteristique in enumerate(caracteristiques):

        plt.annotate(
            caracteristique,
            (
                embeddings_reduits[i, 0],
                embeddings_reduits[i, 1]
            ),
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.title("Visualisation des clusters avec PCA")

    plt.colorbar(label="Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    modele = charger_modele()

    caracteristiques = [
        "Autonomie",
        "Liberté individuelle",
        "Indépendance",
        "Solidarité",
        "Coopération",
        "Entraide",
        "Innovation",
        "Créativité",
        "Progrès technologique"
    ]

    # Plongement lexical
    embeddings = vectoriser_caracteristiques(
        modele,
        caracteristiques
    )

    # Clustering
    modele_kmeans, clusters = appliquer_kmeans(
        embeddings,
        nombre_clusters=3
    )

    print("Dimensions des embeddings :", embeddings.shape)
    print("Clusters :", clusters)

    # Affichage du cluster de chaque caractéristique
    for caracteristique, cluster in zip(caracteristiques, clusters):
        print(f"{caracteristique} : Cluster {cluster}")

    # Méthode du coude
    nombres_clusters, wcss = methode_du_coude(
        embeddings,
        max_clusters=6
    )

    # Affichage des inerties
    for k, inertie in zip(nombres_clusters, wcss):
        print(f"K = {k} | Inertie = {inertie:.4f}")

    # Affichage du graphique
    afficher_methode_du_coude(
        nombres_clusters,
        wcss
    )

    # Coefficient de silhouette
    nombres_clusters, scores = calculer_silhouette(
        embeddings,
        max_clusters=6
    )

    # Affichage des scores
    for k, score in zip(nombres_clusters, scores):
        print(f"K = {k} | Silhouette = {score:.4f}")

    # Meilleur nombre de clusters selon la silhouette
    meilleur_indice = scores.index(max(scores))
    meilleur_k = nombres_clusters[meilleur_indice]

    print("Meilleur K selon la silhouette :", meilleur_k)

    # Affichage du graphique
    afficher_silhouette(
        nombres_clusters,
        scores
    )

    # Clustering avec le meilleur K
    modele_kmeans, clusters = appliquer_kmeans(
        embeddings,
        nombre_clusters=meilleur_k
    )

    # Réduction dimensionnelle
    pca, embeddings_reduits = reduire_dimensions(
        embeddings
    )

    print("Dimensions originales :", embeddings.shape)
    print("Dimensions réduites :", embeddings_reduits.shape)

    # Variance expliquée
    variance = pca.explained_variance_ratio_.sum()

    print(f"Variance expliquée : {variance:.2%}")

    # Affichage des nouveaux clusters
    for caracteristique, cluster in zip(caracteristiques, clusters):
        print(f"{caracteristique} : Cluster {cluster}")

    # Visualisation des clusters
    afficher_clusters(
        embeddings_reduits,
        clusters,
        caracteristiques
    )


