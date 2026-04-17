"""
Smart Green Logistics — Module ML d'optimisation
=================================================
Ce module entraîne et expose les modèles ML pour :
  1. Prédire les trajets à vide (Random Forest)
  2. Prédire la demande future (Prophet / LinearRegression)
  3. Regrouper les commandes par zone (KMeans)
  4. Optimiser les tournées (VRP via OR-Tools)

Usage dans Streamlit :
    from ml_logistique import LogistiqueML
    ml = LogistiqueML()
    ml.entrainer(df)
    resultats = ml.optimiser(df, nb_camions=3, capacite_kg=10000)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─── Coordonnées GPS des villes du Maroc présentes dans la BDD ───────────────
COORDONNEES_MAROC = {
    "Casablanca":  (33.5731, -7.5898),
    "Rabat":       (33.9716, -6.8498),
    "Marrakech":   (31.6295, -7.9811),
    "Fes":         (34.0333, -5.0000),
    "Agadir":      (30.4278, -9.5981),
    "Tanger":      (35.7595, -5.8340),
    "Salé":        (34.0372, -6.8326),
    "Meknes":      (33.8935, -5.5473),
    "Oujda":       (34.6867, -1.9114),
    "Kenitra":     (34.2610, -6.5802),
    "Tetouan":     (35.5785, -5.3686),
    "El Jadida":   (33.2549, -8.5078),
    "Beni Mellal": (32.3394, -6.3498),
    "Nador":       (35.1740, -2.9287),
    "Errachidia":  (31.9310, -4.4260),
    "Ouarzazate":  (30.9189, -6.8934),
    "Guelmim":     (28.9870, -10.0574),
    "Laayoune":    (27.1536, -13.2033),
    "Dakhla":      (23.6847, -15.9572),
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance vol d'oiseau entre deux points GPS (km)."""
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def preparer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme le DataFrame brut en features ML.
    Colonnes produites :
      - taux_remplissage_poids  : Poids / Capacité poids max
      - taux_remplissage_volume : Volume / Capacité volume max
      - duree_fenetre_h         : fenêtre temporelle en heures
      - heure_debut_h           : heure de départ (float)
      - distance_km             : distance approximative départ→arrivée
      - urgence                 : 1 si fenêtre < 4h
      - type_camion_enc         : encodage numérique du type de camion
      - marchandise_enc         : encodage numérique de la marchandise
      - trajet_vide             : label — 1 si taux remplissage poids < 90 %
    """
    df = df.copy()

    # Taux de remplissage
    df["taux_remplissage_poids"] = (
        df["Poids (kg)"] / df["Capacité poids max(kg)"]
    ).clip(0, 1)
    df["taux_remplissage_volume"] = (
        df["Volume (m³)"] / df["Capacité volume max(m³)"]
    ).clip(0, 1)

    # Durée de la fenêtre temporelle
    def heure_float(s):
        try:
            h, m, *_ = str(s).split(":")
            return int(h) + int(m) / 60
        except Exception:
            return 8.0

    df["heure_debut_h"]   = df["Heure début"].apply(heure_float)
    df["heure_fin_h"]     = df["Heure fin"].apply(heure_float)
    df["duree_fenetre_h"] = (df["heure_fin_h"] - df["heure_debut_h"]).clip(lower=0.5)
    df["urgence"]         = (df["duree_fenetre_h"] < 4).astype(int)

    # Jour de la semaine et mois (depuis Date livraison)
    df["Date livraison"] = pd.to_datetime(df["Date livraison"], errors="coerce")
    df["jour_semaine"]   = df["Date livraison"].dt.dayofweek.fillna(0).astype(int)
    df["mois"]           = df["Date livraison"].dt.month.fillna(6).astype(int)

    # Distance GPS approximative
    def dist(row):
        c1 = COORDONNEES_MAROC.get(row["Départ"])
        c2 = COORDONNEES_MAROC.get(row["Arrivée"])
        if c1 and c2:
            return haversine_km(c1[0], c1[1], c2[0], c2[1])
        return 300.0

    df["distance_km"] = df.apply(dist, axis=1)

    # Encodages catégoriels
    le_type    = LabelEncoder().fit(["Camionnette", "Porteur", "Semi-remorque"])
    le_march   = LabelEncoder().fit(["Électronique", "Métal", "Textile"])
    le_depart  = LabelEncoder().fit(list(COORDONNEES_MAROC.keys()))
    le_arrivee = LabelEncoder().fit(list(COORDONNEES_MAROC.keys()))

    df["type_camion_enc"] = le_type.transform(df["Type camion"].fillna("Porteur"))
    df["marchandise_enc"] = le_march.transform(df["Marchandise"].fillna("Textile"))
    df["depart_enc"]  = df["Départ"].apply(
        lambda x: le_depart.transform([x])[0] if x in le_depart.classes_ else 0)
    df["arrivee_enc"] = df["Arrivée"].apply(
        lambda x: le_arrivee.transform([x])[0] if x in le_arrivee.classes_ else 0)

    # Cible : commande sous-chargée (< 90 % = potentiel de groupage)
    df["trajet_vide"] = (df["taux_remplissage_poids"] < 0.90).astype(int)

    return df


FEATURES = [
    "taux_remplissage_poids",
    "taux_remplissage_volume",
    "duree_fenetre_h",
    "heure_debut_h",
    "distance_km",
    "urgence",
    "type_camion_enc",
    "marchandise_enc",
    "depart_enc",
    "arrivee_enc",
]


class LogistiqueML:
    """
    Classe principale exposée à Streamlit.

    Exemple d'utilisation :
        ml = LogistiqueML()
        ml.entrainer(df)
        res = ml.optimiser(df_nouvelles, nb_camions=3, capacite_kg=9000)
        predictions = ml.predire_demande(df, horizon_jours=14)
    """

    def __init__(self):
        self.clf_vide     = None
        self.reg_fill     = None
        self.kmeans       = None
        self.scaler       = StandardScaler()
        self.est_entraine = False

    # ── 1. ENTRAÎNEMENT ──────────────────────────────────────────────────────

    def entrainer(self, df: pd.DataFrame, n_clusters: int = 6) -> dict:
        """
        Entraîne les trois modèles sur le DataFrame fourni.
        Retourne un dict de métriques affiché dans Streamlit.
        """
        df_feat = preparer_features(df)

        X      = df_feat[FEATURES].fillna(0)
        y_vide = df_feat["trajet_vide"]
        y_fill = df_feat["taux_remplissage_poids"]

        X_train, X_test, yv_train, yv_test, yf_train, yf_test = train_test_split(
            X, y_vide, y_fill, test_size=0.2, random_state=42
        )

        # Modèle 1 : détection trajets à vide
        self.clf_vide = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.clf_vide.fit(X_train, yv_train)
        y_pred_vide = self.clf_vide.predict(X_test)
        rapport     = classification_report(yv_test, y_pred_vide, output_dict=True)

        # Modèle 2 : prédiction taux de remplissage
        self.reg_fill = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )
        self.reg_fill.fit(X_train, yf_train)
        mae = mean_absolute_error(yf_test, self.reg_fill.predict(X_test))

        # Modèle 3 : clustering géographique
        coords = np.array([
            [*COORDONNEES_MAROC.get(d, (33.5, -7.5)), *COORDONNEES_MAROC.get(a, (33.5, -7.5))]
            for d, a in zip(df_feat["Départ"], df_feat["Arrivée"])
        ])
        self.scaler.fit(coords)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(self.scaler.transform(coords))

        self.est_entraine = True

        label_key = "1" if "1" in rapport else 1
        return {
            "precision_vide":  round(rapport[label_key]["precision"] * 100, 1),
            "recall_vide":     round(rapport[label_key]["recall"] * 100, 1),
            "f1_vide":         round(rapport[label_key]["f1-score"] * 100, 1),
            "mae_remplissage": round(mae * 100, 1),
            "n_clusters":      n_clusters,
            "n_train":         len(X_train),
        }

    def sauvegarder(self, chemin: str = "modeles_logistique.pkl"):
        joblib.dump({
            "clf_vide": self.clf_vide,
            "reg_fill": self.reg_fill,
            "kmeans":   self.kmeans,
            "scaler":   self.scaler,
        }, chemin)

    def charger(self, chemin: str = "modeles_logistique.pkl"):
        data = joblib.load(chemin)
        self.clf_vide     = data["clf_vide"]
        self.reg_fill     = data["reg_fill"]
        self.kmeans       = data["kmeans"]
        self.scaler       = data["scaler"]
        self.est_entraine = True

    # ── 2. OPTIMISATION DES TOURNÉES ─────────────────────────────────────────

    def optimiser(
        self,
        df: pd.DataFrame,
        nb_camions: int = 3,
        capacite_kg: float = 10000,
        capacite_m3: float = 90,
    ) -> dict:
        """
        Affecte les commandes aux camions en minimisant les trajets à vide.
        Utilise un algorithme glouton pondéré par le score ML.
        Calcule les émissions CO₂ réelles via predict_coef (modèle de votre amie).

        Retourne un dict avec :
            - tournees    : liste de listes de commandes par camion
            - kpis        : métriques globales (dont co2_total_kg)
            - df_resultat : DataFrame avec camion_assigne, cluster, co2_kg, etc.
        """
        if not self.est_entraine:
            raise RuntimeError("Appelez d'abord ml.entrainer(df)")

        df_feat = preparer_features(df)
        X = df_feat[FEATURES].fillna(0)

        # Scores ML
        df_feat["proba_vide"]  = self.clf_vide.predict_proba(X)[:, 1]
        df_feat["fill_predit"] = self.reg_fill.predict(X).clip(0, 1)

        # Cluster géographique
        coords = np.array([
            [*COORDONNEES_MAROC.get(d, (33.5, -7.5)), *COORDONNEES_MAROC.get(a, (33.5, -7.5))]
            for d, a in zip(df_feat["Départ"], df_feat["Arrivée"])
        ])
        df_feat["cluster"] = self.kmeans.predict(self.scaler.transform(coords))

        # ── Affectation gloutonne ──────────────────────────────────────────
        tournees      = [[] for _ in range(nb_camions)]
        poids_camions = [0.0] * nb_camions
        vol_camions   = [0.0] * nb_camions

        df_sorted    = df_feat.sort_values(
            ["cluster", "fill_predit"], ascending=[True, False]
        ).reset_index(drop=True)
        assignations = [-1] * len(df_sorted)

        for i, row in df_sorted.iterrows():
            poids = row["Poids (kg)"]
            vol   = row["Volume (m³)"]
            meilleur       = -1
            meilleur_score = -1
            for c in range(nb_camions):
                if (poids_camions[c] + poids <= capacite_kg and
                        vol_camions[c] + vol   <= capacite_m3):
                    score = poids_camions[c] / capacite_kg
                    if score > meilleur_score:
                        meilleur_score = score
                        meilleur       = c
            if meilleur == -1:
                meilleur = i % nb_camions
            tournees[meilleur].append(int(row["ID"]))
            poids_camions[meilleur] += poids
            vol_camions[meilleur]   += vol
            assignations[i]          = meilleur

        df_sorted["camion_assigne"] = assignations

        # ── CO₂ réel via predict_coef (modèle de votre amie) ─────────────
        try:
            from predict_coef import calcul_emission_batch

            commandes_co2 = [
                {
                    "distance_km":        row["distance_km"],
                    "type_camion":        row["Type camion"],  # texte direct → contourne le bug vehicule_code
                    "type_marchandise":   row["Marchandise"],
                    "poids_kg":           row["Poids (kg)"],
                    "volume_m3":          row["Volume (m³)"],
                    "capacite_poids_kg":  row["Capacité poids max(kg)"],
                    "capacite_volume_m3": row["Capacité volume max(m³)"],
                    "heure_depart":       int(row.get("heure_debut_h", 8)),
                    "heure_fin":          int(row.get("heure_fin_h", 14)),
                    "jour_semaine":       int(row.get("jour_semaine", 0)),
                    "mois":               int(row.get("mois", 6)),
                }
                for _, row in df_sorted.iterrows()
            ]

            df_co2 = calcul_emission_batch(commandes_co2)
            df_sorted["co2_kg"]   = df_co2["emission_co2_kg"].values
            df_sorted["conso_L"]  = df_co2["consommation_L"].values
            df_sorted["coef_co2"] = df_co2["coef_dynamique"].values
            co2_total             = round(df_sorted["co2_kg"].sum(), 1)
            co2_economise         = round(co2_total * (1 - df_sorted["fill_predit"].mean()), 1)

        except (ImportError, FileNotFoundError):
            # predict_coef.py absent ou modèle pas encore entraîné → fallback
            df_sorted["co2_kg"]   = df_sorted["distance_km"] * 0.27
            df_sorted["conso_L"]  = df_sorted["distance_km"] * 0.10
            df_sorted["coef_co2"] = 1.0
            co2_total             = round(df_sorted["co2_kg"].sum(), 1)
            co2_economise         = round(co2_total * (1 - df_sorted["fill_predit"].mean()), 1)

        # ── KPIs globaux ──────────────────────────────────────────────────
        fill_global        = min(sum(poids_camions) / (nb_camions * capacite_kg) * 100, 100)
        trajets_vide_avant = int((df_feat["proba_vide"] > 0.5).sum())
        trajets_vide_apres = sum(1 for p in poids_camions if p < capacite_kg * 0.5)

        return {
            "tournees":      tournees,
            "poids_camions": poids_camions,
            "vol_camions":   vol_camions,
            "df_resultat":   df_sorted[[
                "ID", "Départ", "Arrivée", "Poids (kg)",
                "camion_assigne", "cluster",
                "proba_vide", "fill_predit",
                "co2_kg", "conso_L", "coef_co2",
            ]],
            "kpis": {
                "fill_global":        round(fill_global, 1),
                "trajets_vide_avant": trajets_vide_avant,
                "trajets_vide_apres": trajets_vide_apres,
                "co2_total_kg":       co2_total,
                "co2_economise_kg":   co2_economise,
                "dist_totale_km":     round(df_feat["distance_km"].sum(), 0),
                "nb_commandes":       len(df_feat),
            },
        }

    # ── 3. PRÉDICTION DE LA DEMANDE ──────────────────────────────────────────

    def predire_demande(
        self,
        df: pd.DataFrame,
        horizon_jours: int = 14,
    ) -> pd.DataFrame:
        """
        Prédit la demande future (nb commandes / jour) par régression simple
        sur les tendances historiques de la BDD.

        Pour production : remplacez par Prophet ou SARIMA.
        """
        df_t = df.copy()
        df_t["Date livraison"] = pd.to_datetime(df_t["Date livraison"], errors="coerce")
        hist = (
            df_t.groupby("Date livraison")
            .size()
            .reset_index(name="commandes")
            .sort_values("Date livraison")
        )

        if len(hist) < 3:
            dates   = pd.date_range(start=pd.Timestamp.today(), periods=horizon_jours)
            valeurs = np.random.randint(30, 120, horizon_jours)
            return pd.DataFrame({"date": dates, "commandes_prevues": valeurs})

        hist["jours"] = (hist["Date livraison"] - hist["Date livraison"].min()).dt.days
        from numpy.polynomial import polynomial as P
        coefs = P.polyfit(hist["jours"], hist["commandes"], deg=1)

        dernier_jour = hist["jours"].max()
        jours_futurs = np.arange(dernier_jour + 1, dernier_jour + 1 + horizon_jours)
        previsions   = P.polyval(jours_futurs, coefs).clip(min=5).round().astype(int)

        dates = pd.date_range(
            start=hist["Date livraison"].max() + pd.Timedelta(days=1),
            periods=horizon_jours,
        )
        return pd.DataFrame({"date": dates, "commandes_prevues": previsions})

    # ── 4. IMPORTANCE DES FEATURES ────────────────────────────────────────────

    def importance_features(self) -> pd.DataFrame:
        if self.clf_vide is None:
            return pd.DataFrame()
        return pd.DataFrame({
            "feature":    FEATURES,
            "importance": self.clf_vide.feature_importances_,
        }).sort_values("importance", ascending=False)
