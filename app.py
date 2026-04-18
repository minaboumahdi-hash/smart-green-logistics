import sys, os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & DESIGN SYSTEM (HACKATHON THEME)
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Green Logistics | Hackathon Edition",
    page_icon="assets/logo-removebg-preview.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;700&display=swap');

:root {
    --primary: #2d5a32;
    --secondary: #52a65a;
    --light-bg: #f0faf1;
    --white: #ffffff;
    --danger: #c62828;
    --font-head: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
}

.stApp { background-color: var(--light-bg); color: #1b2e1c; }
* { font-family: var(--font-body); }
h1, h2, h3, h4 { font-family: var(--font-head); color: var(--primary); }

/* Sidebar Premium */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1b5e20 0%, #2e7d32 100%) !important;
}
[data-testid="stSidebar"] * { color: white !important; }

/* Custom Metric Cards */
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    border-left: 5px solid var(--secondary);
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    margin-bottom: 10px;
}
.metric-title { font-size: 0.8rem; text-transform: uppercase; color: #666; font-weight: bold; }
.metric-value { font-size: 1.5rem; font-weight: 800; color: var(--primary); }
.comparison-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
.comp-box { padding: 10px; border-radius: 8px; font-size: 0.85rem; }
.comp-before { background: #fee; border: 1px solid #fcc; }
.comp-after { background: #efe; border: 1px solid #cfc; }

/* Buttons */
.stButton>button {
    background: var(--primary) !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    transition: 0.3s;
    width: 100%;
}
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 10px rgba(0,0,0,0.2); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# DONNÉES & LOGIQUE MÉTIER
# ════════════════════════════════════════════════════════════════════════════
VILLES_MAROC = {
    "Tanger": (35.7595, -5.8340), "Casablanca": (33.5731, -7.5898), "Rabat": (33.9716, -6.8498),
    "Marrakech": (31.6295, -7.9811), "Agadir": (30.4278, -9.5981), "Laayoune": (27.1536, -13.2033),
    "Dakhla": (23.6847, -15.9572), "Laguira": (20.8333, -17.0833)
}

def init_session():
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'role' not in st.session_state: st.session_state.role = None
    if 'reclamations' not in st.session_state: st.session_state.reclamations = []
    if 'history' not in st.session_state: st.session_state.history = []

init_session()

# ════════════════════════════════════════════════════════════════════════════
# COMPOSANTS VISUELS
# ════════════════════════════════════════════════════════════════════════════
def draw_map(points, route=None):
    fig = px.scatter_mapbox(points, lat="lat", lon="lon", zoom=4, height=500)
    if route:
        fig.add_trace(go.Scattermapbox(
            mode="lines+markers",
            lon=[p[1] for p in route], lat=[p[0] for p in route],
            marker={'size': 10}, line={'width': 4, 'color': '#52a65a'}
        ))
    fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
    return fig

def metric_box(label, avant, apres, unit=""):
    reduction = ((avant - apres) / avant * 100) if avant > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{label}</div>
        <div class="comparison-grid">
            <div class="comp-box comp-before">Sans Opti: <b>{avant} {unit}</b></div>
            <div class="comp-box comp-after">Avec Opti: <b>{apres} {unit}</b></div>
        </div>
        <div style="color: #2e7d32; font-size: 0.8rem; margin-top:5px;">Gain de performance: {reduction:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# VUE LOGIN
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("assets/logo-removebg-preview.png", width=200)
        st.markdown("<h2 style='text-align: center;'>Smart Green Logistics</h2>", unsafe_allow_html=True)
        with st.container():
            user = st.text_input("Utilisateur")
            pwd = st.text_input("Mot de passe", type="password")
            if st.button("Se connecter"):
                if user.lower() == "admin":
                    st.session_state.logged_in = True
                    st.session_state.role = "logistique"
                    st.rerun()
                elif user.lower() == "chauffeur":
                    st.session_state.logged_in = True
                    st.session_state.role = "chauffeur"
                    st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# VUE RESPONSABLE LOGISTIQUE
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.role == "logistique":
    with st.sidebar:
        st.image("assets/logo-removebg-preview.png", width=100)
        st.markdown("### Menu Gestion")
        page = st.radio("Navigation", ["Tableau de Bord", "Bilan Journalier", "Gestion Commandes"])
        if st.button("Deconnexion"):
            st.session_state.logged_in = False
            st.rerun()

    st.markdown(f"<h1>Dashboard Logistique Responsable</h1>", unsafe_allow_html=True)
    
    if page == "Tableau de Bord":
        c1, c2, c3 = st.columns(3)
        with c1: metric_box("Trajets a Vide", 450, 120, "km")
        with c2: metric_box("Emissions CO2", 12.5, 8.2, "tonnes")
        with c3: metric_box("Cout Carburant", 15400, 11200, "DH")
        
        c4, c5 = st.columns(2)
        with c4: metric_box("Temps de Tournee", 48, 34, "h")
        with c5:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Taux de Remplissage Quotidien</div>
                <div class="metric-value">88.5 %</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Suivi Temps Reel des Camions (Maroc)")
        # Simulation de points sur la carte
        df_map = pd.DataFrame({'lat': [33.5, 31.6, 23.6], 'lon': [-7.5, -7.9, -15.9], 'camion': ['C1', 'C2', 'C3']})
        st.plotly_chart(draw_map(df_map), use_container_width=True)

    elif page == "Bilan Journalier":
        st.markdown("### Bilan de Performance")
        col_a, col_b = st.columns(2)
        col_a.metric("Commandes Livrees a Temps", "145", "96%")
        col_b.metric("Retards (Pannes/Aléas)", "4", "-2%", delta_color="inverse")
        
        st.markdown("### Liste des Reclamations Chauffeurs")
        if st.session_state.reclamations:
            for rec in st.session_state.reclamations:
                st.warning(f"Chauffeur {rec['id']}: {rec['msg']}")
        else:
            st.info("Aucune reclamation en attente")

    elif page == "Gestion Commandes":
        st.file_uploader("Uploader la liste des commandes (CSV/Excel)")
        st.number_input("Nombre de camions disponibles", value=10)
        st.text_input("Capacite max (Volume m3 / Poids kg)")
        st.button("Lancer l'Optimisation et Assigner les Taches")

# ════════════════════════════════════════════════════════════════════════════
# VUE CHAUFFEUR
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.role == "chauffeur":
    with st.sidebar:
        st.image("assets/logo-removebg-preview.png", width=100)
        st.markdown("### Espace Chauffeur")
        if st.button("Deconnexion"):
            st.session_state.logged_in = False
            st.rerun()

    st.markdown("<h1>Ma Tournee Optimisee</h1>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Feuille de Route", "Signaler un Probleme"])
    
    with tab1:
        st.markdown("### Trajet à suivre")
        route_demo = [VILLES_MAROC["Casablanca"], VILLES_MAROC["Marrakech"], VILLES_MAROC["Agadir"]]
        st.plotly_chart(draw_map(pd.DataFrame([{"lat": r[0], "lon": r[1]} for r in route_demo]), route_demo), use_container_width=True)
        
        st.markdown("### Mes Taches du Jour")
        st.checkbox("Livrer Commande #CMD990 - Marrakech (Client: OCP)")
        st.checkbox("Livrer Commande #CMD992 - Agadir (Client: Agma)")
        st.button("Confirmer la livraison finale")

    with tab2:
        pb_type = st.selectbox("Type de probleme", ["Chemin bloque / Trafic", "Panne Mecanique", "Colis endommage", "Autre"])
        msg = st.text_area("Details du probleme")
        if st.button("Envoyer l'alerte"):
            st.session_state.reclamations.append({"id": "CH001", "msg": f"{pb_type}: {msg}"})
            if "Chemin" in pb_type:
                st.success("Calcul d'une alternative optimale en cours... Veuillez patienter.")
            else:
                st.info("Alerte transmise au responsable logistique. Attendez les instructions.")

# Pied de page Hackathon
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Smart Green Logistics v2.0 - Propulse par l'IA pour un Maroc Durable</p>", unsafe_allow_html=True)
