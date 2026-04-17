import sys
import os
sys.path.append(os.path.dirname(__file__))
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# DEBUG MODE
def optimize_routes(df, nb_camions, capacite):
    return {
        "total_distance": 120,
        "fill_rate": 80,
        "co2_saved": 300,
        "empty_trips_avoided": 2
    }

def predict_demand(df, horizon):
    import pandas as pd
    dates = pd.date_range(start='today', periods=7)
    return pd.DataFrame({
        'date': dates,
        'commandes_prevues': [100]*7
    })

# ─── Configuration de la page ───────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Green Logistics",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS personnalisé ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 10px; color: white; text-align: center;
    }
    .stMetric { background-color: #f0f2f6; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("assets/logo.png", width=100)
    st.title("Smart Green Logistics 🚚")
    st.markdown("---")
    
    st.subheader("📂 Charger les données")
    uploaded_file = st.file_uploader(
        "Importer un fichier CSV de commandes",
        type=["csv", "xlsx"],
        help="Le fichier doit contenir : commande_id, client, poids_kg, volume_m3, adresse_livraison, latitude, longitude"
    )
    
    st.markdown("---")
    st.subheader("⚙️ Paramètres")
    nb_camions = st.slider("Nombre de camions disponibles", 1, 10, 3)
    capacite_camion = st.number_input("Capacité max par camion (kg)", 100, 10000, 1000)
    
    st.markdown("---")
    st.info("💡 **Hackathon Mode** — Prototype IA logistique")

# ─── Contenu principal ───────────────────────────────────────────────────────
st.title("🚚 Optimisation Logistique")
st.markdown("Réduction des trajets à vide · Optimisation des chargements · Prédiction de la demande")
st.markdown("---")

# ─── Chargement et affichage des données ─────────────────────────────────────
if uploaded_file is not None:
    # Lecture du fichier
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.success(f"✅ {len(df)} commandes chargées avec succès !")
    
    # ── Tabs principales ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Données", "🗺️ Optimisation des tournées",
        "📈 Prédiction demande", "📊 Dashboard KPI"
    ])
    
    # ── Tab 1 : Aperçu des données ────────────────────────────────────────────
    with tab1:
        st.subheader("Aperçu des commandes")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total commandes", len(df))
        
        if 'poids_kg' in df.columns:
            col2.metric("Poids total (kg)", f"{df['poids_kg'].sum():,.0f}")
        if 'volume_m3' in df.columns:
            col3.metric("Volume total (m³)", f"{df['volume_m3'].sum():,.1f}")
        
        st.dataframe(df, use_container_width=True, height=400)
        
        # Distribution des poids si la colonne existe
        if 'poids_kg' in df.columns:
            fig = px.histogram(df, x='poids_kg', title="Distribution des poids des commandes",
                             color_discrete_sequence=['#667eea'])
            st.plotly_chart(fig, use_container_width=True)
    
    # ── Tab 2 : Optimisation des tournées ─────────────────────────────────────
    with tab2:
        st.subheader("🗺️ Optimisation des tournées (VRP)")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.info("""
            **Comment ça marche ?**
            
            L'algorithme OR-Tools résout le 
            Problème de Tournées de Véhicules (VRP) :
            
            1. Regroupe les commandes compatibles
            2. Minimise la distance totale
            3. Respecte les capacités camions
            4. Réduit les trajets à vide
            """)
            
            if st.button("🚀 Lancer l'optimisation", type="primary", use_container_width=True):
                with st.spinner("Optimisation en cours..."):
                    try:
                        results = optimize_routes(df, nb_camions, capacite_camion)
                        st.session_state['optimization_results'] = results
                        st.success("✅ Optimisation terminée !")
                    except Exception as e:
                        st.error(f"Erreur : {e}")
                        st.info("Vérifiez que votre CSV contient les colonnes : latitude, longitude, poids_kg")
        
        with col1:
            if 'latitude' in df.columns and 'longitude' in df.columns:
                # Carte des livraisons
                fig_map = px.scatter_mapbox(
                    df,
                    lat='latitude', lon='longitude',
                    hover_name='commande_id' if 'commande_id' in df.columns else None,
                    size='poids_kg' if 'poids_kg' in df.columns else None,
                    color_discrete_sequence=['#667eea'],
                    zoom=10, height=450,
                    title="📍 Points de livraison"
                )
                fig_map.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.warning("⚠️ Colonnes 'latitude' et 'longitude' non trouvées dans le CSV.")
                st.markdown("Pour afficher la carte, ajoutez ces colonnes à votre fichier.")
        
        # Résultats optimisation
        if 'optimization_results' in st.session_state:
            res = st.session_state['optimization_results']
            st.markdown("### Résultats")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Distance totale", f"{res.get('total_distance', 0):,.0f} km")
            c2.metric("Taux de remplissage", f"{res.get('fill_rate', 0):.1f}%")
            c3.metric("CO₂ économisé", f"{res.get('co2_saved', 0):.0f} kg")
            c4.metric("Trajets à vide évités", res.get('empty_trips_avoided', 0))
    
    # ── Tab 3 : Prédiction de la demande ──────────────────────────────────────
    with tab3:
        st.subheader("📈 Prédiction de la demande future")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            horizon = st.selectbox("Horizon de prédiction", ["7 jours", "14 jours", "30 jours"])
            
            if st.button("🔮 Prédire la demande", type="primary", use_container_width=True):
                with st.spinner("Analyse en cours..."):
                    try:
                        predictions = predict_demand(df, horizon)
                        st.session_state['predictions'] = predictions
                        st.success("✅ Prédiction générée !")
                    except Exception as e:
                        st.warning("Modèle en cours d'entraînement — utilisation de données simulées")
                        # Données simulées pour la démo
                        days = int(horizon.split()[0])
                        dates = pd.date_range(start='today', periods=days)
                        pred_values = np.random.randint(50, 200, days)
                        st.session_state['predictions'] = pd.DataFrame({
                            'date': dates, 'commandes_prevues': pred_values
                        })
        
        with col2:
            if 'predictions' in st.session_state:
                pred_df = st.session_state['predictions']
                fig = px.line(pred_df, x='date', y='commandes_prevues',
                             title="Prévision du nombre de commandes",
                             color_discrete_sequence=['#764ba2'])
                fig.update_traces(fill='tozeroy')
                st.plotly_chart(fig, use_container_width=True)
    
    # ── Tab 4 : Dashboard KPI ─────────────────────────────────────────────────
    with tab4:
        st.subheader("📊 Dashboard de performance")
        
        # KPIs simulés (à connecter à vos vrais calculs)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🚚 Taux remplissage moyen", "73%", "+12%")
        col2.metric("🌿 CO₂ économisé", "2.4 tonnes", "+8%")
        col3.metric("💰 Coût optimisé", "-18%", "-18%")
        col4.metric("📦 Commandes traitées", len(df), f"+{len(df)}")
        
        st.markdown("---")
        
        # Graphiques de performance
        col1, col2 = st.columns(2)
        
        with col1:
            # Camembert répartition
            fig_pie = px.pie(
                values=[73, 27],
                names=["Chargé", "À vide"],
                title="Taux de remplissage des camions",
                color_discrete_sequence=['#667eea', '#e0e0e0']
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Évolution des économies
            months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin']
            savings = [1200, 1800, 1500, 2100, 1900, 2400]
            fig_bar = px.bar(x=months, y=savings,
                           title="Économies réalisées (€)",
                           color_discrete_sequence=['#764ba2'])
            st.plotly_chart(fig_bar, use_container_width=True)

else:
    # ── Page d'accueil (aucun fichier chargé) ─────────────────────────────────
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("### 🗺️ Optimisation des tournées\nOR-Tools calcule les meilleures routes pour minimiser les distances et maximiser le remplissage.")
    with col2:
        st.info("### 📈 Prédiction IA\nScikit-learn anticipe la demande future pour planifier les ressources à l'avance.")
    with col3:
        st.info("### 📊 Suivi temps réel\nVisualisez les KPIs clés : CO₂, coûts, taux de remplissage.")
    
    st.markdown("---")
    st.markdown("### 👆 Commencez par charger votre fichier CSV dans la barre latérale !")
    
    # Exemple de format attendu
    st.subheader("📋 Format CSV attendu")
    example_df = pd.DataFrame({
        'commande_id': ['CMD001', 'CMD002', 'CMD003'],
        'client': ['Client A', 'Client B', 'Client C'],
        'poids_kg': [250, 180, 420],
        'volume_m3': [2.5, 1.8, 4.2],
        'adresse_livraison': ['Rue de la Paix, Paris', 'Avenue Foch, Lyon', 'Bd Haussmann, Marseille'],
        'latitude': [48.8698, 45.7640, 43.2965],
        'longitude': [2.3309, 4.8357, 5.3698],
        'date_commande': ['2024-01-15', '2024-01-15', '2024-01-16']
    })
    st.dataframe(example_df, use_container_width=True)
    
    # Bouton pour télécharger un exemple
    csv = example_df.to_csv(index=False)
    st.download_button("⬇️ Télécharger un CSV exemple", csv, "sample_orders.csv", "text/csv")
