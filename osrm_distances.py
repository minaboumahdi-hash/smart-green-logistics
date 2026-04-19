# ── Smart Green Logistics — Recuperation distances reelles via OSRM ───────
# Ce fichier calcule la vraie matrice de distances entre villes marocaines
# en appelant l API OSRM gratuite (pas de cle API necessaire)

import requests
import json

# Coordonnees GPS reelles des villes marocaines
# Format : (latitude, longitude)
VILLES = {
    "Depot (Casa)": (33.5731, -7.5898),
    "Rabat":        (34.0209, -6.8416),
    "Fes":          (34.0181, -5.0078),
    "Marrakech":    (31.6295, -7.9811),
    "Tanger":       (35.7595, -5.8340),
    "Agadir":       (30.4278, -9.5981),
}

def get_matrice_osrm(villes):
    """
    Appelle l API OSRM pour obtenir les vraies distances routieres
    entre toutes les paires de villes
    """
    noms   = list(villes.keys())
    coords = list(villes.values())
    n      = len(coords)

    # Format OSRM : longitude,latitude (attention l ordre est inverse !)
    coords_str = ";".join(
        f"{lon},{lat}" for lat, lon in coords
    )

    url = (
        f"http://router.project-osrm.org/table/v1/driving/"
        f"{coords_str}"
        f"?annotations=distance,duration"
    )

    print("Connexion a OSRM en cours...")
    print(f"Villes : {noms}\n")

    try:
        response = requests.get(url, timeout=15)
        data     = response.json()

        if data.get("code") != "Ok":
            print(f"Erreur OSRM : {data.get('code')}")
            return None, None

        # Distances en metres → convertir en km
        distances_m  = data["distances"]
        distances_km = [
            [round(distances_m[i][j] / 1000, 1) for j in range(n)]
            for i in range(n)
        ]

        # Durees en secondes → convertir en minutes
        durations_s   = data["durations"]
        durations_min = [
            [round(durations_s[i][j] / 60, 0) for j in range(n)]
            for i in range(n)
        ]

        return noms, distances_km, durations_min

    except requests.exceptions.ConnectionError:
        print("Erreur : pas de connexion internet.")
        print("On utilise les distances fictives a la place.")
        return None, None, None

def afficher_matrice(noms, matrice, unite):
    print(f"\nMatrice {unite} :")
    print(f"{'':15}", end="")
    for nom in noms:
        print(f"{nom[:10]:>12}", end="")
    print()
    for i, ligne in enumerate(matrice):
        print(f"{noms[i][:15]:15}", end="")
        for val in ligne:
            print(f"{val:>12}", end="")
        print()

if __name__ == "__main__":
    noms, distances_km, durations_min = get_matrice_osrm(VILLES)

    if distances_km:
        afficher_matrice(noms, distances_km,  "distances (km)")
        afficher_matrice(noms, durations_min, "durees (minutes)")

        # Sauvegarder dans un fichier JSON
        resultat = {
            "villes":       noms,
            "distances_km": distances_km,
            "durees_min":   durations_min
        }
        with open("matrices_maroc.json", "w") as f:
            json.dump(resultat, f, indent=2, ensure_ascii=False)
        print("\nMatrices sauvegardees dans : matrices_maroc.json")
    else:
        print("Impossible de recuperer les donnees OSRM.")