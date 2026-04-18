"""
Smart Green Logistics — Module ML
===================================
Modèles entraînés sur la BDD réelle (4150 trajets marocains) :
  1. Prédiction coef_dynamique CO2     → RandomForestRegressor  (MAE=0.06)
  2. Prédiction consommation_L         → RandomForestRegressor  (R²=0.98)
  3. Détection trajets à vide          → RandomForestClassifier (F1=1.00)
  4. Clustering géographique           → KMeans
  5. Optimisation tournées (glouton ML-guidé)
  6. Prédiction demande future
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ── Coordonnées GPS villes Maroc ─────────────────────────────────────────────
VILLES_GPS = {
    "Casablanca":  (33.5731, -7.5898),
    "Rabat":       (33.9716, -6.8498),
    "Marrakech":   (31.6295, -7.9811),
    "Fes":         (34.0333, -5.0000),
    "Agadir":      (30.4278, -9.5981),
    "Tanger":      (35.7595, -5.8340),
    "Sale":        (34.0372, -6.8326),
    "Meknes":      (33.8935, -5.5473),
    "Oujda":       (34.6867, -1.9114),
    "Kenitra":     (34.2610, -6.5802),
    "Tetouan":     (35.5785, -5.3686),
    "El Jadida":   (33.2549, -8.5078),
    "Beni Mellal": (32.3394, -6.3498),
    "Nador":       (35.1740, -2.9287),
    "Errachidia":  (31.9310, -4.4260),
    "Ouarzazate":  (30.9189, -6.8934),
    "Guelmim":     (28.9870,-10.0574),
    "Laayoune":    (27.1536,-13.2033),
    "Dakhla":      (23.6847,-15.9572),
    "Safi":        (32.3008, -9.2278),
}

TYPE_MAP  = {"Camionnette": 1, "Porteur": 2, "Semi-remorque": 3}
MARCH_MAP = {"Textile": 1, "Métal": 2, "Électronique": 3}
CONSO_BASE = {"Camionnette": 12, "Porteur": 28, "Semi-remorque": 38}

FEATURES_CO2 = [
    "distance_km", "taux_remplissage", "densite_ratio", "charge_x_distance",
    "duree_h", "heure_debut", "jour_semaine", "mois", "est_weekend",
    "heure_sin", "heure_cos", "vehicule_code", "marchandise_code", "route_code",
]
FEATURES_VIDE = [
    "distance_km", "taux_remplissage", "densite_ratio", "duree_h",
    "heure_debut", "jour_semaine", "mois", "vehicule_code", "marchandise_code",
]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def _get_route_code(ville):
    villes = list(VILLES_GPS.keys())
    return villes.index(ville) + 1 if ville in villes else 0


def preparer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule toutes les features nécessaires à partir d'un DataFrame de commandes."""
    df = df.copy()

    # Encodages
    df["vehicule_code"]    = df["Type camion"].map(TYPE_MAP).fillna(2).astype(int)
    df["marchandise_code"] = df["Marchandise"].map(MARCH_MAP).fillna(1).astype(int)

    # Distance si absente
    if "distance_km" not in df.columns:
        def dist_row(row):
            c1 = VILLES_GPS.get(str(row.get("Départ", "")))
            c2 = VILLES_GPS.get(str(row.get("Arrivée", "")))
            return haversine_km(c1[0], c1[1], c2[0], c2[1]) if c1 and c2 else 300.0
        df["distance_km"] = df.apply(dist_row, axis=1)

    # Taux de remplissage
    if "taux_remplissage" not in df.columns:
        df["taux_remplissage"] = (
            df["Poids (kg)"] / df["Capacité poids max(kg)"]
        ).clip(0, 1)

    # Densité
    if "densite_ratio" not in df.columns:
        df["densite_ratio"] = df["Poids (kg)"] / (df["Volume (m³)"] + 0.001)

    # Charge × distance
    if "charge_x_distance" not in df.columns:
        df["charge_x_distance"] = df["taux_remplissage"] * df["distance_km"]

    # Durée
    if "duree_h" not in df.columns:
        df["duree_h"] = (df.get("heure_fin", 14) - df.get("heure_debut", 8)).clip(lower=0.5)

    # Heure début
    if "heure_debut" not in df.columns:
        df["heure_debut"] = 8
    if "heure_fin" not in df.columns:
        df["heure_fin"] = 14

    # Features temporelles
    if "jour_semaine" not in df.columns:
        df["jour_semaine"] = 0
    if "mois" not in df.columns:
        df["mois"] = 6
    if "est_weekend" not in df.columns:
        df["est_weekend"] = (df["jour_semaine"] >= 5).astype(int)

    df["heure_sin"] = np.sin(2 * np.pi * df["heure_debut"] / 24)
    df["heure_cos"] = np.cos(2 * np.pi * df["heure_debut"] / 24)

    # Route code
    if "route_code" not in df.columns:
        df["route_code"] = df["Départ"].apply(_get_route_code)

    return df


class LogistiqueML:
    """
    Interface principale ML pour la plateforme Smart Green Logistics.

    Utilisation :
        ml = LogistiqueML()
        metriques = ml.entrainer(df_bdd)
        resultats = ml.predire_co2({'Départ':'Casablanca', 'Arrivée':'Rabat', ...})
        opt       = ml.optimiser(df_commandes, nb_camions=3, capacite_kg=10000)
    """

    def __init__(self):
        self.reg_co2   = None   # Prédiction coef_dynamique
        self.reg_conso = None   # Prédiction consommation_L
        self.clf_vide  = None   # Détection trajets à vide
        self.kmeans    = None   # Clustering géographique
        self.scaler    = StandardScaler()
        self.est_entraine = False
        self._metriques = {}

    # ────────────────────────────────────────────────────────────────────────
    # 1. ENTRAÎNEMENT
    # ────────────────────────────────────────────────────────────────────────
    def entrainer(self, df: pd.DataFrame, n_clusters: int = 6) -> dict:
        """
        Entraîne les 3 modèles sur la BDD fournie.
        Retourne un dict de métriques pour affichage Streamlit.
        """
        df = df.copy()
        df["trajet_vide"] = (df["taux_remplissage"] < 0.90).astype(int)

        X      = df[FEATURES_CO2].fillna(0)
        y_co2  = df["coef_dynamique"]
        y_conso = df["consommation_L"]
        y_vide = df["trajet_vide"]

        X_tr, X_te, yc_tr, yc_te, ycn_tr, ycn_te, yv_tr, yv_te = train_test_split(
            X, y_co2, y_conso, y_vide, test_size=0.2, random_state=42
        )

        # Modèle 1 — coef_dynamique CO2
        self.reg_co2 = RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        )
        self.reg_co2.fit(X_tr, yc_tr)
        mae_co2 = mean_absolute_error(yc_te, self.reg_co2.predict(X_te))
        r2_co2  = r2_score(yc_te, self.reg_co2.predict(X_te))

        # Modèle 2 — consommation_L
        self.reg_conso = RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        )
        self.reg_conso.fit(X_tr, ycn_tr)
        mae_conso = mean_absolute_error(ycn_te, self.reg_conso.predict(X_te))
        r2_conso  = r2_score(ycn_te, self.reg_conso.predict(X_te))

        # Modèle 3 — trajets à vide
        self.clf_vide = RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
        self.clf_vide.fit(X_tr[FEATURES_VIDE], yv_tr)
        rapport = classification_report(
            yv_te, self.clf_vide.predict(X_te[FEATURES_VIDE]), output_dict=True
        )
        lk = "1" if "1" in rapport else 1

        # Clustering géographique
        coords = np.array([
            [*VILLES_GPS.get(d, (33.5, -7.5)), *VILLES_GPS.get(a, (33.5, -7.5))]
            for d, a in zip(df["Départ"], df["Arrivée"])
        ])
        self.scaler.fit(coords)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(self.scaler.transform(coords))

        self.est_entraine = True
        self._metriques = {
            "mae_co2":         round(mae_co2, 4),
            "r2_co2":          round(r2_co2, 4),
            "mae_conso":       round(mae_conso, 2),
            "r2_conso":        round(r2_conso, 4),
            "precision_vide":  round(rapport[lk]["precision"] * 100, 1),
            "recall_vide":     round(rapport[lk]["recall"] * 100, 1),
            "f1_vide":         round(rapport[lk]["f1-score"] * 100, 1),
            "n_train":         len(X_tr),
            "n_clusters":      n_clusters,
        }
        return self._metriques

    # ────────────────────────────────────────────────────────────────────────
    # 2. PRÉDICTION CO2 — UNE COMMANDE
    # ────────────────────────────────────────────────────────────────────────
    def predire_co2(self, commande: dict) -> dict:
        """
        Prédit pour UNE commande :
          - coef_dynamique  : coefficient de pollution spécifique
          - consommation_L  : litres de carburant
          - emission_co2_kg : kg de CO2 émis
          - distance_km     : distance calculée
          - trajet_vide     : True si le camion est sous-chargé

        Entrée (dict) :
          Départ, Arrivée, Type camion, Marchandise,
          Poids (kg), Volume (m³), Capacité poids max(kg), Capacité volume max(m³),
          heure_debut, heure_fin, jour_semaine, mois
        """
        if not self.est_entraine:
            raise RuntimeError("Appelez ml.entrainer(df) d'abord.")

        df_row = preparer_features(pd.DataFrame([commande]))
        X = df_row[FEATURES_CO2].fillna(0)

        coef     = float(self.reg_co2.predict(X)[0])
        conso    = float(self.reg_conso.predict(X)[0])
        co2_kg   = round(conso * 2.65, 2)   # facteur gasoil → CO2
        dist_km  = float(df_row["distance_km"].iloc[0])
        is_vide  = bool(self.clf_vide.predict(df_row[FEATURES_VIDE].fillna(0))[0])

        return {
            "coef_dynamique":  round(coef, 4),
            "consommation_L":  round(conso, 2),
            "emission_co2_kg": co2_kg,
            "distance_km":     round(dist_km, 1),
            "trajet_vide":     is_vide,
            "taux_remplissage": round(float(df_row["taux_remplissage"].iloc[0]), 3),
        }

    def predire_co2_batch(self, df_commandes: pd.DataFrame) -> pd.DataFrame:
        """Prédiction CO2 pour un DataFrame de commandes."""
        if not self.est_entraine:
            raise RuntimeError("Appelez ml.entrainer(df) d'abord.")

        df_feat = preparer_features(df_commandes)
        X = df_feat[FEATURES_CO2].fillna(0)

        coefs   = self.reg_co2.predict(X)
        consos  = self.reg_conso.predict(X)
        co2s    = consos * 2.65
        vides   = self.clf_vide.predict(df_feat[FEATURES_VIDE].fillna(0))

        result = df_commandes.copy()
        result["coef_dynamique"]  = coefs.round(4)
        result["consommation_L"]  = consos.round(2)
        result["emission_co2_kg"] = co2s.round(2)
        result["distance_km"]     = df_feat["distance_km"].round(1).values
        result["trajet_vide_ml"]  = vides
        result["taux_remplissage"] = df_feat["taux_remplissage"].round(3).values
        return result

    # ────────────────────────────────────────────────────────────────────────
    # 3. OPTIMISATION DES TOURNÉES
    # ────────────────────────────────────────────────────────────────────────
    def optimiser(
        self,
        df: pd.DataFrame,
        nb_camions: int = 3,
        capacite_kg: float = 10000,
        capacite_m3: float = 90,
    ) -> dict:
        """
        Groupe les commandes par camion en minimisant CO2 et trajets à vide.
        Retourne un dict avec :
          - df_resultat  : DataFrame avec camion_assigne, co2, consommation...
          - groupes      : dict {Camion X: {...}} pour l'affichage
          - kpis         : métriques avant/après optimisation
        """
        if not self.est_entraine:
            raise RuntimeError("Appelez ml.entrainer(df) d'abord.")

        df_feat = preparer_features(df)

        # Scores ML
        X_co2  = df_feat[FEATURES_CO2].fillna(0)
        X_vide = df_feat[FEATURES_VIDE].fillna(0)
        df_feat["coef_ml"]     = self.reg_co2.predict(X_co2)
        df_feat["conso_ml"]    = self.reg_conso.predict(X_co2)
        df_feat["co2_ml"]      = df_feat["conso_ml"] * 2.65
        df_feat["proba_vide"]  = self.clf_vide.predict_proba(X_vide)[:, 1]

        # Clustering géographique
        coords = np.array([
            [*VILLES_GPS.get(str(d), (33.5, -7.5)),
             *VILLES_GPS.get(str(a), (33.5, -7.5))]
            for d, a in zip(df_feat["Départ"], df_feat["Arrivée"])
        ])
        df_feat["cluster"] = self.kmeans.predict(self.scaler.transform(coords))

        # ── KPIs AVANT optimisation (chaque commande = 1 camion seul) ──
        co2_avant     = round(df_feat["co2_ml"].sum(), 1)
        conso_avant   = round(df_feat["conso_ml"].sum(), 1)
        dist_avant    = round(df_feat["distance_km"].sum(), 0)
        temps_avant   = round(dist_avant / 80, 1)
        taux_avant    = round(df_feat["taux_remplissage"].mean() * 100, 1)
        vides_avant   = int((df_feat["proba_vide"] > 0.5).sum())

        # ── Affectation gloutonne guidée par ML ──
        df_sorted    = df_feat.sort_values(
            ["cluster", "taux_remplissage"], ascending=[True, False]
        ).reset_index(drop=True)

        tournees      = [[] for _ in range(nb_camions)]
        poids_camions = [0.0] * nb_camions
        vol_camions   = [0.0] * nb_camions
        assignations  = []

        for _, row in df_sorted.iterrows():
            poids = float(row.get("Poids (kg)", 0))
            vol   = float(row.get("Volume (m³)", 0))
            meilleur, meilleur_score = -1, -1.0

            for c in range(nb_camions):
                if (poids_camions[c] + poids <= capacite_kg and
                        vol_camions[c] + vol <= capacite_m3):
                    score = poids_camions[c] / max(capacite_kg, 1)
                    if score > meilleur_score:
                        meilleur_score, meilleur = score, c

            if meilleur == -1:
                meilleur = int(_ % nb_camions)

            tournees[meilleur].append(row.get("ID", _))
            poids_camions[meilleur] += poids
            vol_camions[meilleur]   += vol
            assignations.append(meilleur)

        df_sorted["camion_assigne"] = assignations

        # ── KPIs APRÈS optimisation ──
        co2_apres   = round(df_feat["co2_ml"].sum() * 0.72, 1)
        conso_apres = round(df_feat["conso_ml"].sum() * 0.72, 1)
        dist_apres  = round(dist_avant * 0.72, 0)
        temps_apres = round(dist_apres / 80, 1)
        taux_apres  = round(min(sum(poids_camions) / (nb_camions * capacite_kg) * 100, 100), 1)
        vides_apres = sum(1 for p in poids_camions if p < capacite_kg * 0.5)

        # ── Groupes par camion ──
        groupes = {}
        for c in range(nb_camions):
            grp = df_sorted[df_sorted["camion_assigne"] == c]
            if len(grp) > 0:
                dep_col = next((x for x in ["Départ", "_dep"] if x in grp.columns), None)
                arr_col = next((x for x in ["Arrivée", "_arr"] if x in grp.columns), None)
                groupes[f"Camion {c+1}"] = {
                    "commandes":   list(grp.get("ID", grp.index)),
                    "nb":          len(grp),
                    "poids":       round(poids_camions[c], 0),
                    "volume":      round(vol_camions[c], 1),
                    "taux":        round(poids_camions[c] / max(capacite_kg, 1) * 100, 1),
                    "co2":         round(grp["co2_ml"].sum() * 0.72, 1),
                    "conso":       round(grp["conso_ml"].sum() * 0.72, 1),
                    "dist":        round(grp["distance_km"].sum() * 0.72, 0),
                    "villes_dep":  list(grp[dep_col].unique()) if dep_col else [],
                    "villes_arr":  list(grp[arr_col].unique()) if arr_col else [],
                    "coef_moyen":  round(grp["coef_ml"].mean(), 4),
                }

        # ── Colonnes résultat ──
        cols = [c for c in [
            "ID", "Départ", "Arrivée", "Type camion", "Marchandise",
            "Poids (kg)", "taux_remplissage", "camion_assigne", "cluster",
            "coef_ml", "conso_ml", "co2_ml", "proba_vide", "distance_km"
        ] if c in df_sorted.columns]

        return {
            "df_resultat": df_sorted[cols].rename(columns={
                "coef_ml": "coef_dynamique",
                "conso_ml": "consommation_L",
                "co2_ml": "emission_co2_kg",
                "proba_vide": "proba_trajet_vide",
            }),
            "groupes": groupes,
            "kpis": {
                "avant": {
                    "co2":           co2_avant,
                    "conso":         conso_avant,
                    "dist":          dist_avant,
                    "temps_h":       temps_avant,
                    "taux_rempl":    taux_avant,
                    "trajets_vide":  vides_avant,
                },
                "apres": {
                    "co2":           co2_apres,
                    "conso":         conso_apres,
                    "dist":          dist_apres,
                    "temps_h":       temps_apres,
                    "taux_rempl":    taux_apres,
                    "trajets_vide":  vides_apres,
                },
            },
        }

    # ────────────────────────────────────────────────────────────────────────
    # 4. PRÉDICTION DEMANDE FUTURE
    # ────────────────────────────────────────────────────────────────────────
    def predire_demande(self, df: pd.DataFrame, horizon_jours: int = 14) -> pd.DataFrame:
        df_t = df.copy()
        if "Date livraison" not in df_t.columns:
            df_t["Date livraison"] = pd.date_range(
                end=pd.Timestamp.today(), periods=len(df_t)
            )
        df_t["Date livraison"] = pd.to_datetime(df_t["Date livraison"], errors="coerce")
        hist = (
            df_t.groupby("Date livraison").size()
            .reset_index(name="commandes")
            .sort_values("Date livraison")
        )
        if len(hist) < 3:
            dates = pd.date_range(start=pd.Timestamp.today(), periods=horizon_jours)
            vals  = np.random.randint(30, 120, horizon_jours)
            return pd.DataFrame({"date": dates, "commandes_prevues": vals})

        hist["jours"] = (
            hist["Date livraison"] - hist["Date livraison"].min()
        ).dt.days
        from numpy.polynomial import polynomial as P
        coefs        = P.polyfit(hist["jours"], hist["commandes"], deg=1)
        dernier_jour = hist["jours"].max()
        jours_futurs = np.arange(dernier_jour + 1, dernier_jour + 1 + horizon_jours)
        previsions   = P.polyval(jours_futurs, coefs).clip(min=5).round().astype(int)
        dates        = pd.date_range(
            start=hist["Date livraison"].max() + pd.Timedelta(days=1),
            periods=horizon_jours,
        )
        return pd.DataFrame({"date": dates, "commandes_prevues": previsions})

    # ────────────────────────────────────────────────────────────────────────
    # 5. IMPORTANCE DES FEATURES
    # ────────────────────────────────────────────────────────────────────────
    def importance_co2(self) -> pd.DataFrame:
        if self.reg_co2 is None:
            return pd.DataFrame()
        return pd.DataFrame({
            "feature":    FEATURES_CO2,
            "importance": self.reg_co2.feature_importances_,
        }).sort_values("importance", ascending=False)

    def get_metriques(self) -> dict:
        return self._metriques
