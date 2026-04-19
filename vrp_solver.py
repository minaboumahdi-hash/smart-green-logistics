from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import json
import os
import numpy as np

# =========================================================
# PARAMETRES METIER
# =========================================================
CO2_PAR_LITRE = 2.68          # kg CO2 / litre diesel
CONSO_BASE = 25.0             # L/100 km
CONSO_MAX = 38.0              # L/100 km
PRIX_GASOIL_DH = 11.5         # DH / litre
DEPOT_INDEX_GLOBAL = 0

# =========================================================
# AHP : DETERMINATION DES POIDS
# Critères :
# 0 = Temps
# 1 = Coût carburant
# 2 = CO2
# =========================================================
def ahp_weights(matrix):
    matrix = np.array(matrix, dtype=float)

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_index = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues[max_index].real

    principal_vector = eigenvectors[:, max_index].real
    weights = principal_vector / principal_vector.sum()

    n = matrix.shape[0]
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri_table = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12}
    ri = ri_table.get(n, 1.12)
    cr = ci / ri if ri != 0 else 0.0

    return weights.real, lambda_max, ci, cr


AHP_MATRIX = [
    [1,   2,   1/2],
    [1/2, 1,   1/3],
    [2,   3,   1]
]

AHP_W, LAMBDA_MAX, CI, CR = ahp_weights(AHP_MATRIX)

ALPHA_TEMPS = float(AHP_W[0])
BETA_COUT = float(AHP_W[1])
GAMMA_CO2 = float(AHP_W[2])

# =========================================================
# OUTILS JSON
# =========================================================
def charger_json_obligatoire(nom_fichier):
    if not os.path.exists(nom_fichier):
        raise FileNotFoundError(f"Fichier introuvable : {nom_fichier}")

    with open(nom_fichier, "r", encoding="utf-8") as f:
        contenu = f.read().strip()

    if not contenu:
        raise ValueError(f"Le fichier {nom_fichier} est vide.")

    try:
        return json.loads(contenu)
    except json.JSONDecodeError as e:
        raise ValueError(f"Le fichier {nom_fichier} n'est pas un JSON valide : {e}")


def charger_camions_json():
    noms_possibles = ["camions.json", "camnions.json", "cammions.json", "camion.JSON"]
    for nom in noms_possibles:
        if os.path.exists(nom):
            return nom, charger_json_obligatoire(nom)
    raise FileNotFoundError(
        "Aucun fichier camions trouvé. Noms testés : " + ", ".join(noms_possibles)
    )

# =========================================================
# CHARGEMENT DES DONNEES
# =========================================================
data_osrm = charger_json_obligatoire("matrices_maroc.json")

VILLES = data_osrm["villes"]
DISTANCES_GLOBALES = [[int(float(d)) for d in ligne] for ligne in data_osrm["distances_km"]]
DUREES_GLOBALES = [[int(float(d)) for d in ligne] for ligne in data_osrm["durees_min"]]

def charger_camions():
    nom_fichier, data = charger_camions_json()

    camions = []
    for i, c in enumerate(data):
        camion_id = c.get("id_camion", f"Camion_{i+1}")
        type_camion = c.get("type_camion", "Camion standard")
        capacite_kg = int(float(c["capacite_poids_kg"]))
        capacite_m3 = float(c["capacite_volume_m3"])

        camions.append({
            "id": str(camion_id),
            "type_camion": type_camion,
            "capacite_kg": capacite_kg,
            "capacite_m3": capacite_m3
        })

    if not camions:
        raise ValueError(f"Le fichier {nom_fichier} ne contient aucun camion valide.")

    return camions


def charger_commandes():
    data = charger_json_obligatoire("commandes.json")
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Le fichier commandes.json est vide ou invalide.")
    return data

# =========================================================
# VALIDATION
# =========================================================
def valider_commandes_brutes(commandes, camions):
    if not commandes:
        raise ValueError("Aucune commande fournie.")

    capacite_max_kg = max(c["capacite_kg"] for c in camions)
    capacite_max_m3 = max(c["capacite_m3"] for c in camions)

    for i, cmd in enumerate(commandes, start=1):
        destination = int(cmd["destination"])
        poids = int(cmd["poids_kg"])
        volume = float(cmd["volume_m3"])
        heure_min = int(cmd["heure_min"])
        heure_max = int(cmd["heure_max"])

        if destination == DEPOT_INDEX_GLOBAL:
            raise ValueError(f"Commande {i} invalide : destination = dépôt.")

        if destination < 0 or destination >= len(VILLES):
            raise ValueError(f"Commande {i} invalide : destination hors matrice.")

        if heure_min > heure_max:
            raise ValueError(f"Commande {i} invalide : heure_min > heure_max.")

        if poids > capacite_max_kg:
            raise ValueError(f"Commande {i} trop lourde ({poids} kg) pour la flotte.")

        if volume > capacite_max_m3:
            raise ValueError(f"Commande {i} trop volumineuse ({volume} m3) pour la flotte.")

# =========================================================
# GROUPAGE SIMPLE
# =========================================================
def agreger_commandes_par_destination(commandes):
    agg = {}

    for cmd in commandes:
        dest = int(cmd["destination"])
        poids = int(cmd["poids_kg"])
        volume = float(cmd["volume_m3"])
        heure_min = int(cmd["heure_min"])
        heure_max = int(cmd["heure_max"])

        if dest not in agg:
            agg[dest] = {
                "destination": dest,
                "poids_kg": 0,
                "volume_m3": 0.0,
                "heure_min": heure_min,
                "heure_max": heure_max,
                "nb_commandes": 0,
                "commandes_sources": []
            }

        agg[dest]["poids_kg"] += poids
        agg[dest]["volume_m3"] += volume
        agg[dest]["heure_min"] = max(agg[dest]["heure_min"], heure_min)
        agg[dest]["heure_max"] = min(agg[dest]["heure_max"], heure_max)
        agg[dest]["nb_commandes"] += 1
        agg[dest]["commandes_sources"].append(cmd)

    commandes_agregees = list(agg.values())

    for c in commandes_agregees:
        if c["heure_min"] > c["heure_max"]:
            raise ValueError(
                f"Fenêtres incompatibles après groupage pour la destination {c['destination']}."
            )

    return commandes_agregees

# =========================================================
# SOUS-RESEAU UTILE
# =========================================================
def construire_noeuds_utiles(commandes_agregees):
    return [DEPOT_INDEX_GLOBAL] + sorted({cmd["destination"] for cmd in commandes_agregees})


def extraire_sous_matrice(matrice_globale, noeuds_utiles):
    return [[matrice_globale[i][j] for j in noeuds_utiles] for i in noeuds_utiles]

# =========================================================
# MODELES METIER
# =========================================================
def consommation_litres(distance_km, taux_remplissage):
    conso_l_100 = CONSO_BASE + (CONSO_MAX - CONSO_BASE) * taux_remplissage
    return conso_l_100 * distance_km / 100.0


def calculer_carburant_cout_co2(distance_km, taux_remplissage):
    litres = consommation_litres(distance_km, taux_remplissage)
    cout = litres * PRIX_GASOIL_DH
    co2 = litres * CO2_PAR_LITRE
    return litres, cout, co2


def normaliser_matrice(matrice):
    valeurs = [v for ligne in matrice for v in ligne]
    vmin = min(valeurs)
    vmax = max(valeurs)

    if vmax == vmin:
        return [[0.0 for _ in ligne] for ligne in matrice]

    return [[(v - vmin) / (vmax - vmin) for v in ligne] for ligne in matrice]


def construire_matrices_metier(distances, durees, taux_remplissage_reference=0.5):
    n = len(distances)
    matrice_temps = [[0] * n for _ in range(n)]
    matrice_cout = [[0.0] * n for _ in range(n)]
    matrice_co2 = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            distance_ij = distances[i][j]
            duree_ij = durees[i][j]

            service = 0 if j == 0 else 30
            matrice_temps[i][j] = int(duree_ij + service)

            _, cout, co2 = calculer_carburant_cout_co2(distance_ij, taux_remplissage_reference)
            matrice_cout[i][j] = cout
            matrice_co2[i][j] = co2

    return matrice_temps, matrice_cout, matrice_co2


def construire_cout_agrege(matrice_temps, matrice_cout, matrice_co2):
    temps_norm = normaliser_matrice(matrice_temps)
    cout_norm = normaliser_matrice(matrice_cout)
    co2_norm = normaliser_matrice(matrice_co2)

    n = len(matrice_temps)
    matrice_agregee = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            score = (
                ALPHA_TEMPS * temps_norm[i][j]
                + BETA_COUT * cout_norm[i][j]
                + GAMMA_CO2 * co2_norm[i][j]
            )
            matrice_agregee[i][j] = int(score * 10000)

    return matrice_agregee

# =========================================================
# SCENARIO SANS OPTIMISATION
# =========================================================
def calculer_baseline_sans_optimisation(commandes_agregees):
    litres_total = 0.0
    cout_total = 0.0
    co2_total = 0.0
    distance_totale = 0.0
    temps_total = 0.0
    km_vide = 0.0

    for cmd in commandes_agregees:
        dest = cmd["destination"]
        distance = DISTANCES_GLOBALES[DEPOT_INDEX_GLOBAL][dest] * 2
        duree = DUREES_GLOBALES[DEPOT_INDEX_GLOBAL][dest] * 2 + 30

        distance_totale += distance
        temps_total += duree
        km_vide += DISTANCES_GLOBALES[DEPOT_INDEX_GLOBAL][dest]

        litres, cout, co2 = calculer_carburant_cout_co2(distance, 0.5)
        litres_total += litres
        cout_total += cout
        co2_total += co2

    return {
        "distance_totale_km": round(distance_totale, 2),
        "temps_total_min": int(temps_total),
        "litres_sans_opt": round(litres_total, 2),
        "cout_sans_opt_dh": round(cout_total, 2),
        "co2_sans_opt_kg": round(co2_total, 2),
        "km_vide_sans_opt": round(km_vide, 2),
        "nb_trajets_sans_opt": len(commandes_agregees)
    }

# =========================================================
# OUTILS AFFICHAGE
# =========================================================
def minutes_to_hhmm(minutes):
    h = int(minutes // 60)
    m = int(minutes % 60)
    return f"{h:02d}h{m:02d}"

# =========================================================
# SOLVEUR VRP
# =========================================================
def resoudre_vrp(commandes_brutes, camions):
    valider_commandes_brutes(commandes_brutes, camions)

    commandes_agregees = agreger_commandes_par_destination(commandes_brutes)
    noeuds_utiles = construire_noeuds_utiles(commandes_agregees)

    distances = extraire_sous_matrice(DISTANCES_GLOBALES, noeuds_utiles)
    durees = extraire_sous_matrice(DUREES_GLOBALES, noeuds_utiles)

    matrice_temps, matrice_cout, matrice_co2 = construire_matrices_metier(
        distances, durees, taux_remplissage_reference=0.5
    )
    matrice_cout_agrege = construire_cout_agrege(matrice_temps, matrice_cout, matrice_co2)

    n = len(noeuds_utiles)
    nb_vehicules = len(camions)
    depot_local = 0

    manager = pywrapcp.RoutingIndexManager(n, nb_vehicules, depot_local)
    routing = pywrapcp.RoutingModel(manager)

    def cost_callback(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return matrice_cout_agrege[i][j]

    transit_cost = routing.RegisterTransitCallback(cost_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cost)

    def time_callback(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return matrice_temps[i][j]

    transit_time = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(transit_time, 120, 24 * 60, False, "Temps")
    dim_temps = routing.GetDimensionOrDie("Temps")

    commandes_par_noeud_local = {}
    for cmd in commandes_agregees:
        idx_local = noeuds_utiles.index(cmd["destination"])
        commandes_par_noeud_local[idx_local] = cmd

    for node_local, cmd in commandes_par_noeud_local.items():
        index = manager.NodeToIndex(node_local)
        dim_temps.CumulVar(index).SetRange(cmd["heure_min"], cmd["heure_max"])

    for v in range(nb_vehicules):
        dim_temps.CumulVar(routing.Start(v)).SetRange(6 * 60, 8 * 60)
        dim_temps.CumulVar(routing.End(v)).SetRange(0, 24 * 60)

    def poids_callback(from_index):
        node_local = manager.IndexToNode(from_index)
        if node_local in commandes_par_noeud_local:
            return int(commandes_par_noeud_local[node_local]["poids_kg"])
        return 0

    poids_cb = routing.RegisterUnaryTransitCallback(poids_callback)
    routing.AddDimensionWithVehicleCapacity(
        poids_cb,
        0,
        [c["capacite_kg"] for c in camions],
        True,
        "Poids"
    )

    def volume_callback(from_index):
        node_local = manager.IndexToNode(from_index)
        if node_local in commandes_par_noeud_local:
            return int(commandes_par_noeud_local[node_local]["volume_m3"] * 1000)
        return 0

    volume_cb = routing.RegisterUnaryTransitCallback(volume_callback)
    routing.AddDimensionWithVehicleCapacity(
        volume_cb,
        0,
        [int(c["capacite_m3"] * 1000) for c in camions],
        True,
        "Volume"
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 10

    solution = routing.SolveWithParameters(params)

    if not solution:
        return {"erreur": "Aucune solution trouvée."}

    tournees = []
    litres_total_opt = 0.0
    cout_total_opt = 0.0
    co2_total_opt = 0.0
    distance_totale_opt = 0.0
    temps_total_opt = 0.0
    km_vide_opt = 0.0
    nb_groupages = sum(1 for c in commandes_agregees if c["nb_commandes"] > 1)
    nb_trajets_opt = 0

    for v in range(nb_vehicules):
        index = routing.Start(v)

        arrets = []
        horaires = []
        distance_tournee = 0.0
        temps_tournee = 0.0
        poids_total = 0
        volume_total = 0.0
        nb_commandes_groupees = 0
        last_client_distance_to_depot = 0.0

        while not routing.IsEnd(index):
            node_local = manager.IndexToNode(index)
            node_global = noeuds_utiles[node_local]

            arrets.append(VILLES[node_global])

            heure = solution.Min(dim_temps.CumulVar(index))
            horaires.append(f"{heure // 60:02d}h{heure % 60:02d}")

            if node_local in commandes_par_noeud_local:
                cmd = commandes_par_noeud_local[node_local]
                poids_total += cmd["poids_kg"]
                volume_total += cmd["volume_m3"]
                nb_commandes_groupees += cmd["nb_commandes"]
                last_client_distance_to_depot = DISTANCES_GLOBALES[node_global][DEPOT_INDEX_GLOBAL]

            next_index = solution.Value(routing.NextVar(index))
            next_node_local = manager.IndexToNode(next_index)

            distance_tournee += distances[node_local][next_node_local]
            temps_tournee += matrice_temps[node_local][next_node_local]

            index = next_index

        node_local = manager.IndexToNode(index)
        node_global = noeuds_utiles[node_local]
        arrets.append(VILLES[node_global])

        heure = solution.Min(dim_temps.CumulVar(index))
        horaires.append(f"{heure // 60:02d}h{heure % 60:02d}")

        if len(arrets) > 2 and poids_total > 0:
            nb_trajets_opt += 1

            cap_kg = camions[v]["capacite_kg"]
            cap_m3 = camions[v]["capacite_m3"]

            taux_p = round((poids_total / cap_kg) * 100, 1)
            taux_v = round((volume_total / cap_m3) * 100, 1)
            taux_r = max(taux_p, taux_v) / 100.0

            litres, cout, co2 = calculer_carburant_cout_co2(distance_tournee, taux_r)

            litres_total_opt += litres
            cout_total_opt += cout
            co2_total_opt += co2
            distance_totale_opt += distance_tournee
            temps_total_opt += temps_tournee
            km_vide_opt += last_client_distance_to_depot

            tournees.append({
                "camion": camions[v]["id"],
                "type_camion": camions[v]["type_camion"],
                "arrets": arrets,
                "horaires": horaires,
                "distance_km": round(distance_tournee, 2),
                "temps_total_tournee_min": int(temps_tournee),
                "temps_total_tournee_hhmm": minutes_to_hhmm(temps_tournee),
                "poids_kg": poids_total,
                "volume_m3": round(volume_total, 2),
                "capacite_poids_kg": cap_kg,
                "capacite_volume_m3": cap_m3,
                "taux_remplissage_poids_pct": taux_p,
                "taux_remplissage_volume_pct": taux_v,
                "taux_remplissage_pct": round(taux_r * 100, 1),
                "litres_gasoil": round(litres, 2),
                "cout_carburant_dh": round(cout, 2),
                "co2_kg": round(co2, 2),
                "nb_commandes_groupees": nb_commandes_groupees
            })

    baseline = calculer_baseline_sans_optimisation(commandes_agregees)

    resultat = {
        "parametres_modele": {
            "poids_objectif": {
                "alpha_temps": round(ALPHA_TEMPS, 4),
                "beta_cout": round(BETA_COUT, 4),
                "gamma_co2": round(GAMMA_CO2, 4)
            },
            "ahp": {
                "matrice_comparaison": AHP_MATRIX,
                "lambda_max": round(float(LAMBDA_MAX), 4),
                "CI": round(float(CI), 4),
                "CR": round(float(CR), 4),
                "coherence_valide": bool(CR < 0.10)
            },
            "coefficients_metier": {
                "co2_par_litre": CO2_PAR_LITRE,
                "conso_base_l_100km": CONSO_BASE,
                "conso_max_l_100km": CONSO_MAX,
                "prix_gasoil_dh_l": PRIX_GASOIL_DH
            }
        },
        "entrees": {
            "nb_commandes_brutes": len(commandes_brutes),
            "nb_destinations_apres_groupage": len(commandes_agregees),
            "nb_camions": len(camions),
            "nb_groupages": nb_groupages
        },
        "tournees": tournees,
        "bilan_global": {
            "avec_optimisation": {
                "distance_totale_km": round(distance_totale_opt, 2),
                "temps_total_min": int(temps_total_opt),
                "temps_total_hhmm": minutes_to_hhmm(temps_total_opt),
                "litres_gasoil": round(litres_total_opt, 2),
                "cout_carburant_dh": round(cout_total_opt, 2),
                "co2_kg": round(co2_total_opt, 2),
                "trajets_vide_km": round(km_vide_opt, 2),
                "nb_trajets": nb_trajets_opt
            },
            "sans_optimisation": {
                "distance_totale_km": baseline["distance_totale_km"],
                "temps_total_min": baseline["temps_total_min"],
                "temps_total_hhmm": minutes_to_hhmm(baseline["temps_total_min"]),
                "litres_gasoil": baseline["litres_sans_opt"],
                "cout_carburant_dh": baseline["cout_sans_opt_dh"],
                "co2_kg": baseline["co2_sans_opt_kg"],
                "trajets_vide_km": baseline["km_vide_sans_opt"],
                "nb_trajets": baseline["nb_trajets_sans_opt"]
            },
            "gains": {
                "gain_distance_km": round(baseline["distance_totale_km"] - distance_totale_opt, 2),
                "gain_temps_min": int(baseline["temps_total_min"] - temps_total_opt),
                "gain_litres": round(baseline["litres_sans_opt"] - litres_total_opt, 2),
                "gain_cout_dh": round(baseline["cout_sans_opt_dh"] - cout_total_opt, 2),
                "gain_co2_kg": round(baseline["co2_sans_opt_kg"] - co2_total_opt, 2),
                "gain_trajets_vide_km": round(baseline["km_vide_sans_opt"] - km_vide_opt, 2),
                "gain_nb_trajets": baseline["nb_trajets_sans_opt"] - nb_trajets_opt
            }
        }
    }

    return resultat

# =========================================================
# AFFICHAGE CONSOLE
# =========================================================
def afficher_resultat_console(resultat):
    if "erreur" in resultat:
        print(resultat["erreur"])
        return

    print("=" * 90)
    print("RESULTAT FINAL - VRP MULTI-CRITERES")
    print("=" * 90)

    print("\n1) PARAMETRES DU MODELE")
    print("-" * 90)
    poids = resultat["parametres_modele"]["poids_objectif"]
    ahp = resultat["parametres_modele"]["ahp"]
    metier = resultat["parametres_modele"]["coefficients_metier"]

    print(f"alpha (Temps)  : {poids['alpha_temps']}")
    print(f"beta  (Coût)   : {poids['beta_cout']}")
    print(f"gamma (CO2)    : {poids['gamma_co2']}")
    print(f"CR AHP         : {ahp['CR']}  | Cohérence valide : {ahp['coherence_valide']}")
    print(f"CO2 / litre    : {metier['co2_par_litre']} kg")
    print(f"Conso base     : {metier['conso_base_l_100km']} L/100km")
    print(f"Conso max      : {metier['conso_max_l_100km']} L/100km")
    print(f"Prix gasoil    : {metier['prix_gasoil_dh_l']} DH/L")

    print("\n2) ENTREES PRISES EN COMPTE")
    print("-" * 90)
    entrees = resultat["entrees"]
    print(f"Nombre de commandes brutes           : {entrees['nb_commandes_brutes']}")
    print(f"Destinations après groupage          : {entrees['nb_destinations_apres_groupage']}")
    print(f"Nombre de camions                    : {entrees['nb_camions']}")
    print(f"Nombre de groupages réalisés         : {entrees['nb_groupages']}")

    print("\n3) TOURNEES PAR CAMION")
    print("-" * 90)
    if not resultat["tournees"]:
        print("Aucune tournée exploitable.")
    else:
        for t in resultat["tournees"]:
            print(f"\nCamion {t['camion']} ({t['type_camion']})")
            for arret, heure in zip(t["arrets"], t["horaires"]):
                print(f"  {heure} -> {arret}")
            print(f"  Distance totale                    : {t['distance_km']} km")
            print(f"  Temps total tournée                : {t['temps_total_tournee_hhmm']} ({t['temps_total_tournee_min']} min)")
            print(f"  Poids total                        : {t['poids_kg']} kg")
            print(f"  Volume total                       : {t['volume_m3']} m3")
            print(f"  Remplissage poids                  : {t['taux_remplissage_poids_pct']} %")
            print(f"  Remplissage volume                 : {t['taux_remplissage_volume_pct']} %")
            print(f"  Remplissage retenu                 : {t['taux_remplissage_pct']} %")
            print(f"  Carburant                          : {t['litres_gasoil']} L")
            print(f"  Coût carburant                     : {t['cout_carburant_dh']} DH")
            print(f"  CO2                                : {t['co2_kg']} kg")
            print(f"  Commandes groupées                 : {t['nb_commandes_groupees']}")

    print("\n4) BILAN GLOBAL")
    print("-" * 90)
    avec_opt = resultat["bilan_global"]["avec_optimisation"]
    sans_opt = resultat["bilan_global"]["sans_optimisation"]
    gains = resultat["bilan_global"]["gains"]

    print("\nAvec optimisation :")
    for k, v in avec_opt.items():
        print(f"  {k:<35} : {v}")

    print("\nSans optimisation :")
    for k, v in sans_opt.items():
        print(f"  {k:<35} : {v}")

    print("\nGains obtenus :")
    for k, v in gains.items():
        print(f"  {k:<35} : {v}")

# =========================================================
# EXECUTION
# =========================================================
if __name__ == "__main__":
    try:
        camions = charger_camions()
        commandes_brutes = charger_commandes()

        resultat = resoudre_vrp(commandes_brutes, camions)
        afficher_resultat_console(resultat)

        with open("resultat_solver.json", "w", encoding="utf-8") as f:
            json.dump(resultat, f, indent=2, ensure_ascii=False)

        print("\nJSON sauvegardé dans : resultat_solver.json")

    except Exception as e:
        print(f"\nERREUR : {e}")
        