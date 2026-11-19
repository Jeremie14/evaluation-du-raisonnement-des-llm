import json
from pathlib import Path
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


# retourne la liste d'articles
# chaque articles est dictionnaire avec les clés
# dict_keys(['dossier_ids', 'essay_id', 'language_code', 'raw_text', 'source_author', 'source_category', 'source_published_at', 'source_updated_at', 'title', 'url'])
def extractionDonnees(cheminFichier):
    with open(cheminFichier / "development.json", "r", encoding="utf-8") as fichier:
        development = json.load(fichier)
    articles = development["essays"]
    return articles

def nettoyer_html(html):
    soup = BeautifulSoup(html, "html.parser")

    # Remplacer les balises structurantes par des séparateurs
    for balise in soup.find_all([
        "p", "div", "br", "h1", "h2", "h3",
        "h4", "h5", "h6", "li", "blockquote"
    ]):
        balise.append("\n")

    texte = soup.get_text()

    # Nettoyer les espaces
    texte = re.sub(r"[ \t]+", " ", texte)
    texte = re.sub(r"\n\s*\n+", "\n\n", texte)

    return texte.strip()

def nettoyage(articles):
    for article in articles:
        article["clean_text"] = nettoyer_html(
            article.get("raw_text", "")
        )
    return articles

def validationArticles(articles):
    articles_valides = [
        a for a in articles
        if a.get("clean_text", "").strip()
    ]

    articles_exclus = [
        a for a in articles
        if not a.get("clean_text", "").strip()
    ]
    return articles_valides, articles_exclus

def evaluationDistributionLongueur(articles):

    longueurs = np.array([
        len(a["clean_text"].split())
        for a in articles
    ])

    print("Nombre d'articles :", len(longueurs))
    print("Minimum :", longueurs.min(), "mots")
    print("Maximum :", longueurs.max(), "mots")
    print("Moyenne :", round(longueurs.mean(), 1), "mots")
    print("Médiane :", int(np.median(longueurs)))
    print("Écart-type :", round(longueurs.std(), 1))


if __name__ == '__main__':
    racine = Path.cwd().parent.parent  # fonctionne avec notebook
    chemin = racine / "projet" / "donnees_experimentales"

    articles = extractionDonnees(chemin)
    articles = nettoyage(articles)
    print(articles[0].keys())


    articles_vides = [
        a for a in articles
        if not a.get("clean_text", "").strip()
    ]

    print("Nombre d'articles concernés :", len(articles_vides))

    articles_valides, articles_exclus = validationArticles(articles)
    print("valide :", len(articles_valides), "exclus :",len(articles_exclus))

    evaluationDistributionLongueur(articles)