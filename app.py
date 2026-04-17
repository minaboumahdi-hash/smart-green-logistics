import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


@st.cache_data
def load_users():
    return pd.read_csv("data/users.csv", dtype=str)


def optimize_routes(df, nb_camions, capacite):
    total_poids = df['poids_kg'].sum() if 'poids_kg' in df.columns else 0
    capacite_totale = nb_camions * capacite
    fill_rate = min((total_poids / capacite_totale) * 100, 100) if capacite_totale > 0 else 0
    return {
        "total_distance": 342,
        "fill_rate": round(fill_rate, 1),
        "co2_saved": round(fill_rate * 0.8, 1),
        "empty_trips_avoided": nb_camions - 1
    }


def predict_demand(df, horizon):
    days = int(horizon.split()[0])
    dates = pd.date_range(start='today', periods=days)
    return pd.DataFrame({'date': dates, 'commandes_prevues': np.random.randint(50, 200, days)})


st.set_page_config(
    page_title="Smart Green Logistics",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stTextInput input { border-radius: 8px; }
    .stButton button {
        width: 100%;
        border-radius: 8px;
        background-color: #2e7d32 !important;
        color: white !important;
        font-weight: bold;
        border: none !important;
    }
    .stButton button:hover {
        background-color: #1b5e20 !important;
    }
</style>
""", unsafe_allow_html=True)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None


def show_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.image("assets/logo-removebg-preview.png", width=200)
        st.markdown("---")
        st.subheader("🔐 Connexion")
        with st.form("login_form"):
            user_id = st.text_input("👤 ID Utilisateur", placeholder="Ex: 1")
            password = st.text_input("🔑 Mot de passe", type="password", placeholder="Votre mot de passe")
            submit = st.form_submit_button("Se connecter", use_container_width=True)
            if submit:
                users = load_users()
                match = users[(users['id'] == user_id) & (users['password'] == password)]
                if not match.empty:
                    st.session_state.logged_in = True
                    st.session_state.user = match.iloc[0].to_dict()
                    st.rerun()
                else:
                    st.error("❌ ID ou mot de passe incorrect")
        st.markdown("---")
        st.caption("💡 Contactez votre responsable pour obtenir vos identifiants")


def show_chauffeur():
    user = st.session_state.user

    with st.sidebar:
        st.image("assets/logo-removebg-preview.png", width=150)
        st.markdown(f"### 👋 Bonjour, {user['nom']}")
        st.markdown("**Rôle :** 🚛 Chauffeur")
        st.markdown("---")
        if st.button("🚪 Se déconnecter", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()

    st.title(f"🚛 Espace Chauffeur — {user['nom']}")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Mes tournées",
        "🗺️ Ma route",
        "✅ Livraisons",
        "⚠️ Signaler un problème"
    ])

    with tab1:
        st.subheader("📋 Mes tournées assignées")
        tournees = pd.DataFrame({
            'Commande': ['CMD001', 'CMD002', 'CMD003'],
            'Client': ['Client A', 'Client B', 'Client C'],
            'Adresse': ['Casablanca, Maarif', 'Rabat, Agdal', 'Salé, Centre'],
            'Poids (kg)': [250, 180, 420],
            'Statut': ['En attente', 'En attente', 'En attente']
        })
        st.dataframe(tournees, use_container_width=True)
        st.metric("Total livraisons aujourd'hui", 3)

    with tab2:
        st.subheader("🗺️ Ma route du jour")
        points = pd.DataFrame({
            'lat': [33.5731, 33.9716, 34.0209],
            'lon': [-7.5898, -6.8498, -6.8416],
            'lieu': ['Casablanca', 'Rabat', 'Salé']
        })
        fig = px.scatter_mapbox(
            points, lat='lat', lon='lon', hover_name='lieu',
            zoom=8, height=400,
            color_discrete_sequence=['#2e7d32']
        )
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("✅ Marquer une livraison comme terminée")
        commande = st.selectbox("Sélectionner la commande", ['CMD001', 'CMD002', 'CMD003'])
        note = st.text_area("Note (optionnel)", placeholder="Ex: Client absent, livraison au voisin...")
        if st.button("✅ Confirmer la livraison", type="primary"):
            st.success(f"✅ Livraison {commande} marquée comme terminée !")
            st.balloons()

    with tab4:
        st.subheader("⚠️ Signaler un problème")
        type_probleme = st.selectbox("Type de problème", [
            "🚗 Panne véhicule",
            "📦 Colis endommagé",
            "🚦 Embouteillage / Retard",
            "📍 Adresse introuvable",
            "Autre"
        ])
        description = st.text_area("Description", placeholder="Décrivez le problème...")
        urgence = st.radio("Niveau d'urgence", ["🟢 Faible", "🟡 Moyen", "🔴 Urgent"])
        if st.button("📤 Envoyer le signalement", type="primary"):
            st.success("✅ Signalement envoyé au responsable !")


def show_responsable():
    user = st.session_state.user

    with st.sidebar:
        st.image("assets/logo-removebg-preview.png", width=150)
        st.markdown(f"### 👋 Bonjour, {user['nom']}")
        st.markdown("**Rôle :** 📊 Responsable Logistique")
        st.markdown("---")
        st.subheader("📂 Charger les données")
        uploaded_file = st.file_uploader("Importer CSV de commandes", type=["csv", "xlsx"])
        st.markdown("---")
        st.subheader("⚙️ Paramètres")
        nb_camions = st.slider("Nombre de camions", 1, 10, 3)
        capacite_camion = st.number_input("Capacité max (kg)", 100, 10000, 1000)
        st.markdown("---")
        if st.button("🚪 Se déconnecter", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()

    st.title(f"📊 Espace Responsable — {user['nom']}")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Toutes les tournées",
        "🗺️ Optimisation IA",
        "👥 Gestion chauffeurs",
        "📈 Prédiction demande",
        "📊 Dashboard KPI"
    ])

    with tab1:
        st.subheader("📋 Toutes les tournées en cours")
        tournees_all = pd.DataFrame({
            'Commande': ['CMD001', 'CMD002', 'CMD003', 'CMD004'],
            'Chauffeur': ['Ahmed', 'Mohamed', 'Karim', 'Non assigné'],
            'Client': ['Client A', 'Client B', 'Client C', 'Client D'],
            'Statut': ['En cours', 'En attente', 'Terminé', 'Non assigné'],
            'Poids (kg)': [250, 180, 420, 300]
        })
        st.dataframe(tournees_all, use_container_width=True)

    with tab2:
        st.subheader("🗺️ Optimisation des tournées (VRP)")
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success(f"✅ {len(df)} commandes chargées !")
            col1, col2 = st.columns([2, 1])
            with col2:
                if st.button("🚀 Lancer l'optimisation", type="primary", use_container_width=True):
                    with st.spinner("Optimisation en cours..."):
                        results = optimize_routes(df, nb_camions, capacite_camion)
                        st.session_state['optimization_results'] = results
                        st.success("✅ Optimisation terminée !")
            with col1:
                if 'latitude' in df.columns and 'longitude' in df.columns:
                    fig_map = px.scatter_mapbox(
                        df, lat='latitude', lon='longitude',
                        zoom=10, height=400,
                        color_discrete_sequence=['#2e7d32']
                    )
                    fig_map.update_layout(mapbox_style="open-street-map")
                    st.plotly_chart(fig_map, use_container_width=True)
            if 'optimization_results' in st.session_state:
                res = st.session_state['optimization_results']
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Distance totale", f"{res['total_distance']:,} km")
                c2.metric("Taux remplissage", f"{res['fill_rate']:.1f}%")
                c3.metric("CO₂ économisé", f"{res['co2_saved']:.0f} kg")
                c4.metric("Trajets vide évités", res['empty_trips_avoided'])
        else:
            st.info("👆 Chargez un fichier CSV dans la barre latérale")

    with tab3:
        st.subheader("👥 Gestion des chauffeurs")
        chauffeurs = pd.DataFrame({
            'Nom': ['Ahmed', 'Mohamed', 'Karim'],
            'Statut': ['🟢 En route', '🟡 En attente', '🟢 En route'],
            "Livraisons aujourd'hui": [3, 1, 4],
            'Taux ponctualité': ['95%', '88%', '92%']
        })
        st.dataframe(chauffeurs, use_container_width=True)
        st.markdown("### Assigner une tournée")
        col1, col2 = st.columns(2)
        with col1:
            chauffeur_sel = st.selectbox("Chauffeur", ['Ahmed', 'Mohamed', 'Karim'])
        with col2:
            commande_sel = st.selectbox("Commande", ['CMD004', 'CMD005'])
        if st.button("✅ Assigner", type="primary"):
            st.success(f"✅ Commande {commande_sel} assignée à {chauffeur_sel} !")

    with tab4:
        st.subheader("📈 Prédiction de la demande")
        col1, col2 = st.columns([1, 2])
        with col1:
            horizon = st.selectbox("Horizon", ["7 jours", "14 jours", "30 jours"])
            if st.button("🔮 Prédire", type="primary"):
                predictions = predict_demand(pd.DataFrame(), horizon)
                st.session_state['predictions'] = predictions
                st.success("✅ Prédiction générée !")
        with col2:
            if 'predictions' in st.session_state:
                fig = px.line(
                    st.session_state['predictions'],
                    x='date',
                    y='commandes_prevues',
                    color_discrete_sequence=['#2e7d32']
                )
                fig.update_traces(fill='tozeroy')
                st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("📊 Dashboard KPI")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🚚 Taux remplissage", "73%", "+12%")
        c2.metric("🌿 CO₂ économisé", "2.4 tonnes", "+8%")
        c3.metric("💰 Coût optimisé", "-18%", "-18%")
        c4.metric("📦 Chauffeurs actifs", "3/3")
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(
                values=[73, 27],
                names=["Chargé", "À vide"],
                color_discrete_sequence=['#2e7d32', '#e0e0e0']
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            fig_bar = px.bar(
                x=['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin'],
                y=[1200, 1800, 1500, 2100, 1900, 2400],
                color_discrete_sequence=['#2e7d32']
            )
            st.plotly_chart(fig_bar, use_container_width=True)


if not st.session_state.logged_in:
    show_login()
else:
    role = st.session_state.user['role']
    if role == 'chauffeur':
        show_chauffeur()
    elif role == 'responsable':
        show_responsable()
    else:
        st.error("Rôle non reconnu")
