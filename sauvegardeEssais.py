
import json
from pathlib import Path
from datetime import datetime

def sauvegarder_analyses(analyses, dossier, descriptif):
    """
    Sauvegarde les analyses dans un nouveau fichier JSON.
    Ne remplace jamais un fichier existant.
    """

    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)

    # Préparer les métadonnées
    donnees = {
        "metadata": {
            "descriptif": descriptif,
            "modele": "Qwen/Qwen2.5-1.5B-Instruct",
            "version_prompt": 1,
            "date_creation": datetime.now().isoformat(),
            "nombre_articles": len(analyses)
        },
        "analyses": analyses
    }

    # Générer un nom horodaté
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
    nom_base = f"analyses_qwen_{horodatage}"

    compteur = 0

    while True:
        suffixe = "" if compteur == 0 else f"_{compteur}"
        chemin = dossier / f"{nom_base}{suffixe}.json"

        try:
            with open(chemin, "x", encoding="utf-8") as fichier:
                json.dump(
                    donnees,
                    fichier,
                    ensure_ascii=False,
                    indent=4
                )
            break

        except FileExistsError:
            compteur += 1

    print(f"Fichier sauvegardé : {chemin}")
    print(f"Nombre d'analyses : {len(analyses)}")

    return chemin
