"""
ml_integration.py
==================
Fonctions d'intégration ML pour app.py :
  - get_ml()               : singleton LogistiqueML dans la session
  - carte_maroc_complete() : carte Plotly du Maroc entier
  - tab_optimisation()     : onglet Optimisation IA
  - tab_prediction()       : onglet Prédiction demande
  - tab_carte_chauffeur()  : carte pour l'espace chauffeur
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ─── Import direct — le sys.path est déjà configuré par app.py ───────────────
from ml_logistique import LogistiqueML, COORDONNEES_MAROC

VILLES_MAROC = {
    "Tanger":      (35.7595, -5.8340),
    "Tetouan":     (35.5785, -5.3686),
    "Al Hoceima":  (35.2517, -3.9372),
    "Nador":       (35.1740, -2.9287),
    "Oujda":       (34.6867, -1.9114),
    "Kenitra":     (34.2610, -6.5802),
    "Rabat":       (33.9716, -6.8498),
    "Salé":        (34.0372, -6.8326),
    "Casablanca":  (33.5731, -7.5898),
    "Mohammedia":  (33.6861, -7.3830),
    "Meknes":      (33.8935, -5.5473),
    "Fes":         (34.0333, -5.0000),
    "Ifrane":      (33.5228, -5.1118),
    "Khenifra":    (32.9342, -5.6681),
    "El Jadida":   (33.2549, -8.5078),
    "Safi":        (32.3007, -9.2278),
    "Marrakech":   (31.6295, -7.9811),
    "Beni Mellal": (32.3394, -6.3498),
    "Khouribga":   (32.8810, -6.9071),
    "Settat":      (33.0017, -7.6194),
    "Essaouira":   (31.5085, -9.7595),
    "Agadir":      (30.4278, -9.5981),
    "Taroudant":   (30.4738, -8.8752),
    "Tiznit":      (29.6974, -9.7316),
    "Ouarzazate":  (30.9189, -6.8934),
    "Errachidia":  (31.9310, -4.4260),
    "Zagora":      (30.3286, -5.8384),
    "Guelmim":     (28.9870, -10.0574),
    "Tan-Tan":     (28.4378, -11.1028),
    "Laayoune":    (27.1536, -13.2033),
    "Smara":       (26.7333, -11.6750),
    "Dakhla":      (23.6847, -15.9572),
}

COULEURS_CLUSTER = ["#2e7d32","#1565c0","#e65100","#6a1b9a","#00838f","#c62828"]


def get_ml() -> LogistiqueML:
    if "ml_model" not in st.session_state:
        st.session_state.ml_model = LogistiqueML()
        if os.path.exists("modeles_logistique.pkl"):
            try:
                st.session_state.ml_model.charger("modeles_logistique.pkl")
            except Exception:
                pass
    return st.session_state.ml_model


def carte_maroc_complete(df_commandes=None, clusters=None, titre="Carte logistique — Maroc"):
    villes_df = pd.DataFrame([
        {"ville": v, "lat": c[0], "lon": c[1]} for v, c in VILLES_MAROC.items()
    ])
    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=villes_df["lat"], lon=villes_df["lon"],
        mode="markers+text",
        marker=dict(size=7, color="#9e9e9e", opacity=0.7),
        text=villes_df["ville"], textposition="top right",
        textfont=dict(size=9, color="#555555"),
        name="Villes", hovertemplate="<b>%{text}</b><extra></extra>",
    ))
    if df_commandes is not None and "Départ" in df_commandes.columns:
        for _, row in df_commandes.iterrows():
            dep = COORDONNEES_MAROC.get(row["Départ"])
            arr = COORDONNEES_MAROC.get(row["Arrivée"])
            if dep and arr:
                cluster_idx = int(row.get("cluster", 0)) % len(COULEURS_CLUSTER)
                couleur = COULEURS_CLUSTER[cluster_idx] if clusters is not None else "#2e7d32"
                fig.add_trace(go.Scattermapbox(
                    lat=[dep[0], arr[0]], lon=[dep[1], arr[1]],
                    mode="lines", line=dict(width=1.5, color=couleur),
                    opacity=0.5, showlegend=False, hoverinfo="skip",
                ))
        df_dep = df_commandes.copy()
        df_dep["lat_dep"] = df_dep["Départ"].map(lambda x: COORDONNEES_MAROC.get(x, (33.5,-7.5))[0])
        df_dep["lon_dep"] = df_dep["Départ"].map(lambda x: COORDONNEES_MAROC.get(x, (33.5,-7.5))[1])
        fig.add_trace(go.Scattermapbox(
            lat=df_dep["lat_dep"], lon=df_dep["lon_dep"], mode="markers",
            marker=dict(size=10, opacity=0.9,
                color=df_dep.get("cluster", pd.Series(0, index=df_dep.index)).apply(
                    lambda x: COULEURS_CLUSTER[int(x) % len(COULEURS_CLUSTER)])),
            text=df_dep["Départ"] + " → " + df_dep["Arrivée"],
            name="Commandes", hovertemplate="<b>%{text}</b><extra></extra>",
        ))
    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=30.0, lon=-7.0), zoom=4.8),
        margin=dict(l=0, r=0, t=30, b=0), height=500,
        title=dict(text=titre, font=dict(size=14)),
        legend=dict(orientation="h", y=-0.05),
    )
    return fig


def tab_optimisation(df, nb_camions, capacite_camion):
    st.subheader("🗺️ Optimisation des tournées par IA")
    ml = get_ml()
    col_carte, col_actions = st.columns([3, 1])

    with col_actions:
        st.markdown("#### Paramètres")
        st.slider("Seuil sous-chargement (%)", 50, 95, 90, step=5)
        n_clusters = st.slider("Zones géographiques", 2, 8, 6)
        if st.button("🧠 (Re)Entraîner le modèle", use_container_width=True):
            with st.spinner("Entraînement en cours..."):
                metriques = ml.entrainer(df, n_clusters=n_clusters)
                ml.sauvegarder("modeles_logistique.pkl")
                st.session_state["metriques_ml"] = metriques
            st.success("✅ Modèle entraîné et sauvegardé !")
        if "metriques_ml" in st.session_state:
            m = st.session_state["metriques_ml"]
            st.markdown("**Métriques modèle :**")
            st.metric("F1 (trajets à vide)", f"{m['f1_vide']}%")
            st.metric("MAE remplissage",     f"{m['mae_remplissage']}%")
        st.markdown("---")
        btn_opt = st.button("🚀 Optimiser les tournées", type="primary",
                            use_container_width=True, disabled=not ml.est_entraine)
        if not ml.est_entraine:
            st.caption("⚠️ Entraînez d'abord le modèle")

    with col_carte:
        if btn_opt and ml.est_entraine:
            with st.spinner("Optimisation ML en cours..."):
                res = ml.optimiser(df, nb_camions=nb_camions, capacite_kg=capacite_camion)
                st.session_state["opt_results"] = res
        df_carte = None
        if "opt_results" in st.session_state:
            df_res = st.session_state["opt_results"]["df_resultat"]
            df_carte = df_res if "Départ" in df_res.columns else \
                df_res.merge(df[["ID", "Départ", "Arrivée"]], on="ID", how="left")
        fig_carte = carte_maroc_complete(
            df_commandes=df_carte, clusters=df_carte is not None,
            titre="Carte des tournées optimisées — Maroc complet",
        )
        st.plotly_chart(fig_carte, use_container_width=True)

    if "opt_results" in st.session_state:
        kpis     = st.session_state["opt_results"]["kpis"]
        tournees = st.session_state["opt_results"]["tournees"]
        poids    = st.session_state["opt_results"]["poids_camions"]
        st.markdown("### 📊 Résultats de l'optimisation")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🚚 Remplissage global",     f"{kpis['fill_global']}%")
        c2.metric("⚠️ Trajets à vide (avant)", kpis["trajets_vide_avant"])
        c3.metric("✅ Trajets à vide (après)", kpis["trajets_vide_apres"],
                  delta=f"-{kpis['trajets_vide_avant'] - kpis['trajets_vide_apres']}",
                  delta_color="inverse")
        c4.metric("🌿 CO₂ économisé",          f"{int(kpis['co2_economise_kg'])} kg")
        c5.metric("📦 Commandes traitées",      kpis["nb_commandes"])
        st.markdown("#### Répartition par camion")
        st.dataframe(pd.DataFrame({
            "Camion":           [f"🚛 Camion {i+1}" for i in range(nb_camions)],
            "Commandes":        [len(t) for t in tournees],
            "Poids total (kg)": [round(p, 0) for p in poids],
            "Taux remplissage": [f"{min(p/capacite_camion*100,100):.1f}%" for p in poids],
        }), use_container_width=True, hide_index=True)
        df_res = st.session_state["opt_results"]["df_resultat"]
        for i in range(nb_camions):
            with st.expander(f"🚛 Détail Camion {i+1} — {len(tournees[i])} livraisons"):
                df_cam = df_res[df_res["camion_assigne"] == i][[
                    "ID", "Départ", "Arrivée", "Poids (kg)",
                    "cluster", "proba_vide", "fill_predit", "co2_kg",
                ]].copy()
                df_cam["proba_vide"]  = df_cam["proba_vide"].apply(lambda x: f"{x*100:.1f}%")
                df_cam["fill_predit"] = df_cam["fill_predit"].apply(lambda x: f"{x*100:.1f}%")
                df_cam["co2_kg"]      = df_cam["co2_kg"].apply(lambda x: f"{x:.1f} kg")
                df_cam.columns = ["ID","Départ","Arrivée","Poids (kg)","Zone","Risque vide","Fill prédit","CO₂"]
                st.dataframe(df_cam, use_container_width=True, hide_index=True)
    if ml.est_entraine:
        with st.expander("🔬 Importance des variables (Random Forest)"):
            imp_df = ml.importance_features()
            fig_imp = px.bar(imp_df.head(8), x="importance", y="feature", orientation="h",
                             color_discrete_sequence=["#2e7d32"],
                             labels={"importance": "Importance", "feature": "Variable"})
            fig_imp.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_imp, use_container_width=True)


def tab_prediction(df):
    st.subheader("📈 Prédiction de la demande")
    ml = get_ml()
    col1, col2 = st.columns([1, 2])
    with col1:
        horizon       = st.selectbox("Horizon", ["7 jours", "14 jours", "30 jours"])
        horizon_jours = int(horizon.split()[0])
        marchandise_filtre = st.multiselect(
            "Filtrer par marchandise",
            options=df["Marchandise"].unique().tolist(),
            default=df["Marchandise"].unique().tolist(),
        )
        if st.button("🔮 Générer la prédiction", type="primary"):
            df_filtre   = df[df["Marchandise"].isin(marchandise_filtre)]
            predictions = ml.predire_demande(df_filtre, horizon_jours=horizon_jours)
            st.session_state["predictions"] = predictions
            st.success(f"✅ Prédiction sur {horizon_jours} jours générée !")
    with col2:
        if "predictions" in st.session_state:
            pred = st.session_state["predictions"]
            fig  = px.line(pred, x="date", y="commandes_prevues",
                           color_discrete_sequence=["#2e7d32"],
                           labels={"commandes_prevues": "Commandes prévues", "date": "Date"})
            fig.update_traces(fill="tozeroy", fillcolor="rgba(46,125,50,0.1)")
            fig.add_hline(y=pred["commandes_prevues"].mean(), line_dash="dot",
                          line_color="#e65100",
                          annotation_text=f"Moy: {pred['commandes_prevues'].mean():.0f}")
            fig.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Pic prévu", f"{pred['commandes_prevues'].max()}")
            c2.metric("Moyenne",   f"{pred['commandes_prevues'].mean():.0f}")
            c3.metric("Total",     f"{pred['commandes_prevues'].sum()}")


def tab_carte_chauffeur(tournees_chauffeur):
    st.subheader("🗺️ Ma route du jour — Maroc")
    fig = carte_maroc_complete(titre="Ma route — Carte du Maroc")
    for i, t in enumerate(tournees_chauffeur):
        dep = COORDONNEES_MAROC.get(t["depart"])
        arr = COORDONNEES_MAROC.get(t["arrivee"])
        if dep and arr:
            fig.add_trace(go.Scattermapbox(
                lat=[dep[0], arr[0]], lon=[dep[1], arr[1]],
                mode="lines+markers",
                line=dict(width=4, color="#2e7d32"),
                marker=dict(size=12, color=["#1565c0", "#c62828"]),
                name=f"Stop {i+1}: {t['depart']} → {t['arrivee']}",
                text=[t["depart"], t["arrivee"]],
                hovertemplate="<b>%{text}</b><extra></extra>",
            ))
    st.plotly_chart(fig, use_container_width=True)
