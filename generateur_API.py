import os
import re
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

with open("questions.json", "r", encoding="utf-8") as f:
    QUESTIONS = json.load(f)

CONSIGNE_FORMAT = "Rédige en texte brut continu, sans titre, sans liste à puces et sans formatage Markdown."

CONDITIONS = {
    "baseline_neutre": (
        f"Réponds à cette question de manière approfondie, en environ 350 mots. "
        f"{CONSIGNE_FORMAT} Question : {{question}}"
    ),
    "analyse_contradictoire": (
        f"Analyse cette question de manière approfondie en environ 350 mots. "
        f"Ne cherche pas à fournir une réponse définitive ou consensuelle : "
        f"explore la pluralité des perspectives, les arguments contradictoires, "
        f"les présupposés sous-jacents, les arbitrages et les angles morts, "
        f"afin de donner au lecteur les éléments pour construire son propre jugement. "
        f"{CONSIGNE_FORMAT} Question : {{question}}"
    ),
    "amorce_extrait": (
        f"Voici un court extrait de réflexion sur le sujet : « {{extrait}} ». "
        f"À partir de cette mise en contexte, réponds à la question suivante de manière approfondie, en environ 850 mots. "
        f"{CONSIGNE_FORMAT} Question : {{question}}"
    )
}


def nettoyer_markdown(texte: str) -> str:
    if not texte or texte.startswith("ERREUR:"):
        return texte

    texte = re.sub(r"```.*?```", "", texte, flags=re.DOTALL)
    texte = re.sub(r"^\s*#{1,6}\s+", "", texte, flags=re.MULTILINE)
    texte = re.sub(r"(\*\*|__)(.*?)\1", r"\2", texte)
    texte = re.sub(r"(\*|_)(.*?)\1", r"\2", texte)
    texte = re.sub(r"^\s*[-*+]\s+", "", texte, flags=re.MULTILINE)
    texte = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", texte)
    texte = re.sub(r"`([^`]+)`", r"\1", texte)

    texte = re.sub(r"[ \t]+", " ", texte)
    texte = re.sub(r"\n\s*\n+", "\n\n", texte)
    return texte.strip()


class GenerateurGemini:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Clé GEMINI_API_KEY introuvable.")

        self.client = genai.Client(api_key=api_key)
        self.modele = 'gemini-3.8-flash'

    def generer_texte_persistant(self, prompt: str, parametres: dict = None) -> str:
        """Réessaie jusqu'à obtenir un texte valide."""
        if parametres is None:
            parametres = {"temperature": 0.7}

        config = types.GenerateContentConfig(**parametres)
        tentative = 0

        while True:
            tentative += 1
            try:
                reponse = self.client.models.generate_content(
                    model=self.modele,
                    contents=prompt,
                    config=config
                )
                if reponse.text and reponse.text.strip():
                    return reponse.text
                raise ValueError("Réponse vide reçue de l'API.")

            except Exception as e:
                erreur = str(e)
                # Erreurs fatales : inutile de boucler à l'infini
                if any(code in erreur for code in ["400", "401", "403", "404", "API_KEY_INVALID"]):
                    print(f"    [X] Erreur fatale : {erreur}")
                    return f"ERREUR: {erreur}"

                # Backoff progressif : 30s, 60s, 120s, 240s, puis plafond à 300s (5 min)
                pause = min(30 * (2 ** min(tentative - 1, 4)), 300)
                print(f"    [!] Échec temporaire (tentative {tentative}) : {erreur[:80]}... Pause de {pause}s.")
                time.sleep(pause)


def lancer_experience():
    generateur = GenerateurGemini()
    fichier_json = "textes_generes_gemini.json"

    # Reprise automatique : on recharge les textes déjà réussis
    resultats_par_id = {}
    if os.path.exists(fichier_json):
        with open(fichier_json, "r", encoding="utf-8") as f:
            anciens = json.load(f)
            for r in anciens:
                if not r.get("clean_text", "").startswith("ERREUR:"):
                    resultats_par_id[r["essay_id"]] = r
        print(f"=== Reprise : {len(resultats_par_id)} textes déjà valides conservés ===")

    total_cible = len(QUESTIONS) * len(CONDITIONS)
    print(f"=== Objectif : {total_cible} textes ({len(QUESTIONS)} questions x {len(CONDITIONS)} conditions) ===")

    for i, item in enumerate(QUESTIONS):
        question = item["question"] if isinstance(item, dict) else item
        extrait = item.get("extrait", "") if isinstance(item, dict) else ""

        print(f"\nQuestion {i + 1}/{len(QUESTIONS)} : {question[:60]}...")

        for nom_condition, gabarit in CONDITIONS.items():
            essay_id = f"gemini_q{i + 1}_{nom_condition}"

            # On saute si déjà généré avec succès
            if essay_id in resultats_par_id:
                print(f"  -> {nom_condition} : déjà fait, on passe.")
                continue

            # On saute amorce_extrait s'il n'y a pas d'extrait fourni
            if nom_condition == "amorce_extrait" and not extrait:
                print(f"  -> {nom_condition} : aucun extrait fourni, ignoré.")
                continue

            print(f"  -> Condition : {nom_condition} (génération en cours...)")
            prompt_final = gabarit.format(question=question, extrait=extrait)

            texte_genere = generateur.generer_texte_persistant(prompt_final)
            texte_propre = nettoyer_markdown(texte_genere)

            resultats_par_id[essay_id] = {
                "essay_id": essay_id,
                "question": question,
                "modele": generateur.modele,
                "prompt": prompt_final,
                "condition": nom_condition,
                "parametres": {"temperature": 0.7},
                "clean_text": texte_propre
            }

            # Sauvegarde immédiate après chaque succès
            with open(fichier_json, "w", encoding="utf-8") as f:
                json.dump(list(resultats_par_id.values()), f, ensure_ascii=False, indent=2)

            print(f"    [OK] Sauvegardé ({len(resultats_par_id)}/{total_cible})")
            time.sleep(5)

    print(f"\n=== Terminé ! {len(resultats_par_id)}/{total_cible} textes valides dans '{fichier_json}' ===")


if __name__ == "__main__":
    lancer_experience()