"""
Solveur VRP avec OR-Tools + AHP multi-objectif (Temps, Coût, CO2).
Version refactorisée : importable comme module depuis app.py.
"""
import json
import os
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ============================================================
# 1. AHP — Calcul des poids α (temps), β (coût), γ (CO2)
# ============================================================
def calculer_poids_ahp(matrice_comparaison=None):
    """
    Calcule les poids AHP par méthode des valeurs propres.
    matrice_comparaison : matrice 3x3 de Saaty. Par défaut : équilibre
    privilégiant légèrement le coût et le CO2.
    Retourne : (poids dict, ratio_coherence)
    """
    if matrice_comparaison is None:
        # Par défaut : Coût > CO2 > Temps (à ajuster selon ton contexte)
        matrice_comparaison = np.array([
            [1,     1/2,  1/3],   # Temps
            [2,     1,    1/2],   # Coût
            [3,     2,    1  ],   # CO2
        ])
    M = np.array(matrice_comparaison, dtype=float)
    valeurs_propres, vecteurs_propres = np.linalg.eig(M)
    idx_max = np.argmax(valeurs_propres.real)
    lambda_max = valeurs_propres[idx_max].real
    vecteur = vecteurs_propres[:, idx_max].real
    poids = vecteur / vecteur.sum()
    n = M.shape[0]
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0
    RI = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12}.get(n, 1.24)
    CR = CI / RI if RI > 0 else 0
    return {
        "alpha_temps": float(poids[0]),
        "beta_cout":   float(poids[1]),
        "gamma_co2":   float(poids[2]),
        "CR": float(CR),
        "coherent": bool(CR < 0.1),
    }

# ============================================================
# 2. Construction de la matrice de coût pondérée AHP
# ============================================================
def construire_matrice_cout(distances_km, poids_ahp,
                            cout_par_km=2.5, co2_par_km=0.9, vitesse_kmh=60):
    """
    Combine distance/temps/coût/CO2 en une seule matrice de coût
    entière (OR-Tools exige des entiers).
    """
    n = len(distances_km)
    matrice = np.zeros((n, n), dtype=int)
    a = poids_ahp["alpha_temps"]
    b = poids_ahp["beta_cout"]
    g = poids_ahp["gamma_co2"]
    for i in range(n):
        for j in range(n):
            d = distances_km[i][j]
            temps_min = (d / vitesse_kmh) * 60
            cout_dh   = d * cout_par_km
            co2_kg    = d * co2_par_km
            score = a * temps_min + b * cout_dh + g * co2_kg
            matrice[i][j] = int(score * 100)  # *100 pour la précision
    return matrice.tolist()

# ============================================================
# 3. Solveur VRP principal
# ============================================================
def resoudre_vrp(commandes, camions, matrice_distances_km,
                 noms_villes=None, matrice_ahp=None):
    """
    Résout le problème VRP avec capacités (poids + volume).

    PARAMETRES
    ----------
    commandes : liste de dicts. Chaque dict doit contenir :
        { "id": str, "ville_index": int, "poids": float, "volume": float }
        ville_index = indice de la ville dans matrice_distances_km
        (l'index 0 est réservé au DEPOT)
    camions : liste de dicts. Chaque dict doit contenir :
        { "id": str, "capacite_poids": float, "capacite_volume": float }
    matrice_distances_km : matrice carrée NxN des distances en km.
        Index 0 = dépôt, 1..N-1 = villes des commandes.
    noms_villes : liste optionnelle des noms ["Depot", "Casa", ...]
    matrice_ahp : matrice 3x3 Saaty optionnelle pour l'AHP

    RETOUR
    ------
    dict {
      "poids_ahp": {...},
      "tournees": [
        { "camion_id": ..., "trajet": [...], "distance_km": ...,
          "poids_total": ..., "volume_total": ..., "co2_kg": ... }
      ],
      "total_distance_km": ...,
      "total_co2_kg": ...,
      "commandes_non_servies": [...]
    }
    """
    # ---- 1) AHP
    poids_ahp = calculer_poids_ahp(matrice_ahp)

    # ---- 2) Matrice de coût pondérée
    matrice_cout = construire_matrice_cout(matrice_distances_km, poids_ahp)

    # ---- 3) Demandes (poids + volume) par nœud
    n_noeuds = len(matrice_distances_km)
    demandes_poids  = [0] * n_noeuds
    demandes_volume = [0] * n_noeuds
    for cmd in commandes:
        idx = cmd["ville_index"]
        demandes_poids[idx]  += int(cmd["poids"] * 1000)   # en grammes
        demandes_volume[idx] += int(cmd["volume"] * 1000)  # en litres

    # ---- 4) Capacités camions
    capacites_poids  = [int(c["capacite_poids"]  * 1000) for c in camions]
    capacites_volume = [int(c["capacite_volume"] * 1000) for c in camions]
    n_camions = len(camions)

    # ---- 5) Modèle OR-Tools
    manager = pywrapcp.RoutingIndexManager(n_noeuds, n_camions, 0)  # depot=0
    routing = pywrapcp.RoutingModel(manager)

    def cout_callback(from_idx, to_idx):
        i = manager.IndexToNode(from_idx)
        j = manager.IndexToNode(to_idx)
        return matrice_cout[i][j]
    transit_idx = routing.RegisterTransitCallback(cout_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # Contrainte poids
    def demande_poids_cb(from_idx):
        return demandes_poids[manager.IndexToNode(from_idx)]
    poids_cb_idx = routing.RegisterUnaryTransitCallback(demande_poids_cb)
    routing.AddDimensionWithVehicleCapacity(
        poids_cb_idx, 0, capacites_poids, True, "Poids")

    # Contrainte volume
    def demande_volume_cb(from_idx):
        return demandes_volume[manager.IndexToNode(from_idx)]
    vol_cb_idx = routing.RegisterUnaryTransitCallback(demande_volume_cb)
    routing.AddDimensionWithVehicleCapacity(
        vol_cb_idx, 0, capacites_volume, True, "Volume")

    # Pénalité de non-visite (rend les nœuds optionnels en cas d'infaisabilité)
    penalite = 10_000_000
    for node in range(1, n_noeuds):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalite)

    # ---- 6) Paramètres de recherche
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.seconds = 10

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return {
            "poids_ahp": poids_ahp,
            "tournees": [],
            "total_distance_km": 0,
            "total_co2_kg": 0,
            "commandes_non_servies": [c["id"] for c in commandes],
            "erreur": "Aucune solution trouvée",
        }

    # ---- 7) Extraction des tournées
    tournees = []
    total_dist = 0
    villes_servies = set()
    for v in range(n_camions):
        index = routing.Start(v)
        trajet, dist_km, p_tot, v_tot = [], 0, 0, 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            nom = noms_villes[node] if noms_villes else f"Noeud_{node}"
            trajet.append({"index": node, "nom": nom})
            villes_servies.add(node)
            p_tot += demandes_poids[node]
            v_tot += demandes_volume[node]
            next_idx = solution.Value(routing.NextVar(index))
            if not routing.IsEnd(next_idx):
                dist_km += matrice_distances_km[node][manager.IndexToNode(next_idx)]
            index = next_idx
        # retour dépôt
        node_fin = manager.IndexToNode(index)
        nom_fin = noms_villes[node_fin] if noms_villes else "Depot"
        trajet.append({"index": node_fin, "nom": nom_fin})

        if len(trajet) > 2:  # camion utilisé
            tournees.append({
                "camion_id": camions[v]["id"],
                "trajet": trajet,
                "distance_km": round(dist_km, 2),
                "poids_total_kg": p_tot / 1000,
                "volume_total_m3": v_tot / 1000,
                "co2_kg": round(dist_km * 0.9, 2),
            })
            total_dist += dist_km

    # Commandes non servies
    non_servies = [c["id"] for c in commandes if c["ville_index"] not in villes_servies]

    return {
        "poids_ahp": poids_ahp,
        "tournees": tournees,
        "total_distance_km": round(total_dist, 2),
        "total_co2_kg": round(total_dist * 0.9, 2),
        "nb_camions_utilises": len(tournees),
        "commandes_non_servies": non_servies,
    }


# ============================================================
# 4. Mode standalone : lit les JSON et affiche le résultat
# ============================================================
if __name__ == "__main__":
    base = os.path.dirname(__file__)
    with open(os.path.join(base, "matrices_maroc.json")) as f:
        matrices = json.load(f)
    with open(os.path.join(base, "camions.json")) as f:
        camions = json.load(f)
    with open(os.path.join(base, "commandes.json")) as f:
        commandes = json.load(f)

    distances = matrices.get("distances_km") or matrices["distances"]
    villes    = matrices.get("villes", [])

    res = resoudre_vrp(commandes, camions, distances, noms_villes=villes)
    print(json.dumps(res, indent=2, ensure_ascii=False))

