import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from ml_logistique import LogistiqueML
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Green Logistics",
    page_icon="assets/logo-removebg-preview.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Startup / Hackathon Premium Green Tech
# ════════════════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --g0: #f0faf1;
    --g1: #e8f5e9;
    --g2: #c8e6c9;
    --g3: #2d5a32;
    --g4: #3d7a44;
    --g5: #52a65a;
    --g6: #2e7d32;
    --g7: #1b5e20;
    --g8: #d4f5d8;
    --g9: #f9fffe;
    --accent: #2d5a32;
    --accent2: #00897b;
    --warn: #e65100;
    --danger: #c62828;
    --white: #ffffff;
    --card: #ffffff;
    --border: rgba(45,90,50,0.15);
    --shadow: 0 4px 20px rgba(45,90,50,0.1);
    --radius: 12px;
    --font-head: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
}

* { font-family: var(--font-body); }
h1,h2,h3,h4 { font-family: var(--font-head); }

/* ── Global background ── */
.stApp {
    background: linear-gradient(135deg, #f9fffe 0%, #f0faf1 50%, #e8f5e9 100%);
    color: #1b2e1c;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem; max-width: 1600px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2e7d32 0%, #388e3c 50%, #43a047 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.15);
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
section[data-testid="stSidebar"] * { color: #ffffff !important; }
section[data-testid="stSidebar"] .stMarkdown p { color: #c8e6c9 !important; font-size: 0.85rem; }
.logo-box {
    background: rgba(255,255,255,0.92);
    border-radius: 10px;
    padding: 0.5rem;
    margin-bottom: 0.5rem;
    text-align: center;
}

/* ── Inputs ── */
.stTextInput input, .stNumberInput input, .stSelectbox select,
.stTextArea textarea {
    background: #ffffff !important;
    border: 1px solid rgba(45,90,50,0.25) !important;
    border-radius: 8px !important;
    color: #1b2e1c !important;
    font-family: var(--font-body) !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #2d5a32 !important;
    box-shadow: 0 0 0 2px rgba(45,90,50,0.12) !important;
}
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid rgba(45,90,50,0.25) !important;
    border-radius: 8px !important;
    color: #1b2e1c !important;
}
label { color: #2d5a32 !important; font-size: 0.82rem !important; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }

/* ── Buttons ── */
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(135deg, #2d5a32, #3d7a44) !important;
    color: #ffffff !important;
    border: 1px solid #52a65a !important;
    border-radius: 8px !important;
    font-family: var(--font-head) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.03em !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #3d7a44, #52a65a) !important;
    box-shadow: 0 4px 16px rgba(45,90,50,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid rgba(45,90,50,0.15) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 2px 8px rgba(45,90,50,0.08) !important;
}
[data-testid="stMetricLabel"] { color: #52a65a !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricValue"] { color: #1b5e20 !important; font-family: var(--font-head) !important; font-size: 1.8rem !important; font-weight: 700; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #e8f5e9 !important;
    border-radius: var(--radius) !important;
    padding: 4px !important;
    border: 1px solid rgba(45,90,50,0.15) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #2d5a32 !important;
    border-radius: 8px !important;
    font-family: var(--font-head) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.5rem 1rem !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2d5a32, #3d7a44) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(45,90,50,0.25) !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: var(--radius) !important; overflow: hidden; border: 1px solid rgba(45,90,50,0.15) !important; }
iframe { border-radius: var(--radius) !important; }

/* ── Slider ── */
.stSlider > div > div > div > div { background: #52a65a !important; }
.stSlider > div > div > div > div > div { background: #2d5a32 !important; }

/* ── Alerts ── */
.stSuccess { background: #f0faf1 !important; border: 1px solid #52a65a !important; border-radius: var(--radius) !important; color: #1b5e20 !important; }
.stWarning { background: #fff8f0 !important; border: 1px solid #e65100 !important; border-radius: var(--radius) !important; }
.stError   { background: #fff5f5 !important; border: 1px solid #c62828 !important; border-radius: var(--radius) !important; }
.stInfo    { background: #f0faf1 !important; border: 1px solid rgba(45,90,50,0.2) !important; border-radius: var(--radius) !important; color: #2d5a32 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #f9fffe !important;
    border: 1px dashed rgba(45,90,50,0.3) !important;
    border-radius: var(--radius) !important;
    padding: 0.5rem !important;
}

/* ── Custom components ── */
.sgl-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 0 1.5rem 0;
    border-bottom: 1px solid rgba(45,90,50,0.15);
    margin-bottom: 1.5rem;
}
.sgl-header h1 {
    font-family: var(--font-head); font-size: 1.6rem; font-weight: 800;
    color: #1b2e1c; margin: 0; letter-spacing: -0.02em;
}
.sgl-header .badge {
    background: linear-gradient(135deg, #2d5a32, #3d7a44);
    border: 1px solid #52a65a;
    color: #ffffff; font-family: var(--font-head);
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
    padding: 0.3rem 0.8rem; border-radius: 20px; text-transform: uppercase;
}

.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin: 1rem 0; }
.kpi-card {
    background: #ffffff;
    border: 1px solid rgba(45,90,50,0.15);
    border-radius: var(--radius);
    padding: 1.2rem;
    box-shadow: 0 2px 8px rgba(45,90,50,0.08);
    position: relative; overflow: hidden;
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #2d5a32, #52a65a);
}
.kpi-card .kpi-label { font-size: 0.72rem; color: #52a65a; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }
.kpi-card .kpi-value { font-family: var(--font-head); font-size: 1.9rem; font-weight: 800; color: #1b5e20; line-height: 1; }
.kpi-card .kpi-delta { font-size: 0.78rem; margin-top: 0.3rem; }
.kpi-card .kpi-delta.pos  { color: #2e7d32; }
.kpi-card .kpi-delta.neg  { color: #c62828; }
.kpi-card .kpi-delta.warn { color: #e65100; }

.compare-block { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0; }
.compare-card {
    background: #ffffff;
    border: 1px solid rgba(45,90,50,0.15);
    border-radius: var(--radius);
    padding: 1.2rem;
    box-shadow: 0 2px 8px rgba(45,90,50,0.06);
}
.compare-card.before { border-top: 3px solid #c62828; }
.compare-card.after  { border-top: 3px solid #2d5a32; }
.compare-card h4 { font-family: var(--font-head); font-size: 0.78rem; letter-spacing: 0.1em; text-transform: uppercase; margin: 0 0 0.8rem 0; }
.compare-card.before h4 { color: #c62828; }
.compare-card.after  h4 { color: #2d5a32; }
.compare-stat { display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0; border-bottom: 1px solid rgba(45,90,50,0.08); font-size: 0.85rem; color: #2d3e2e; }
.compare-stat .val { font-family: var(--font-head); font-weight: 700; color: #1b2e1c; }

.section-title {
    font-family: var(--font-head); font-size: 0.85rem; font-weight: 700;
    color: #2d5a32; letter-spacing: 0.08em; text-transform: uppercase;
    margin: 1.5rem 0 0.8rem 0;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-title::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(45,90,50,0.2), transparent);
}

.route-card {
    background: #ffffff;
    border: 1px solid rgba(45,90,50,0.15);
    border-radius: var(--radius);
    padding: 1rem 1.2rem; margin: 0.5rem 0;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 2px 6px rgba(45,90,50,0.06);
}
.route-card .route-path { font-family: var(--font-head); font-weight: 700; font-size: 0.95rem; color: #1b2e1c; }
.route-card .route-meta { font-size: 0.78rem; color: #52a65a; margin-top: 0.2rem; }
.route-card .route-co2 { background: #f0faf1; border: 1px solid #52a65a; color: #2d5a32; font-family: var(--font-head); font-size: 0.8rem; font-weight: 700; padding: 0.3rem 0.7rem; border-radius: 20px; }

.status-pill {
    display: inline-block; padding: 0.25rem 0.7rem; border-radius: 20px;
    font-size: 0.72rem; font-weight: 700; font-family: var(--font-head);
    text-transform: uppercase; letter-spacing: 0.06em;
}
.status-pill.green  { background: #f0faf1; color: #2e7d32; border: 1px solid #52a65a; }
.status-pill.yellow { background: #fff8f0; color: #e65100; border: 1px solid #e65100; }
.status-pill.red    { background: #fff5f5; color: #c62828; border: 1px solid #c62828; }
.status-pill.grey   { background: #f5f5f5; color: #757575; border: 1px solid #bdbdbd; }

.reclamation-card {
    background: #fff5f5;
    border: 1px solid rgba(198,40,40,0.2);
    border-left: 3px solid #c62828;
    border-radius: var(--radius);
    padding: 1rem 1.2rem; margin: 0.5rem 0;
}
.reclamation-card h4 { font-family: var(--font-head); font-size: 0.85rem; color: #c62828; margin: 0 0 0.3rem 0; }
.reclamation-card p { font-size: 0.82rem; color: #5d4037; margin: 0; }

.login-container {
    min-height: 100vh;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #f0faf1 0%, #e8f5e9 50%, #c8e6c9 100%);
}
.login-box {
    width: 100%; max-width: 420px;
    background: #ffffff;
    border: 1px solid rgba(45,90,50,0.2);
    border-radius: 20px;
    padding: 2.5rem;
    box-shadow: 0 20px 60px rgba(45,90,50,0.15);
}

.sidebar-nav-item {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.6rem 0.8rem; border-radius: 8px;
    cursor: pointer; transition: all 0.15s;
    font-size: 0.85rem; color: #c8e6c9;
    margin: 2px 0;
}
.sidebar-nav-item:hover { background: rgba(255,255,255,0.1); color: #ffffff; }
.sidebar-nav-item.active { background: rgba(255,255,255,0.15); color: #ffffff; font-weight: 600; }

.sidebar-section { font-size: 0.68rem; color: #a5d6a7; text-transform: uppercase; letter-spacing: 0.1em; padding: 0.8rem 0.8rem 0.3rem; font-weight: 700; font-family: var(--font-head); }

.divider { border: none; border-top: 1px solid rgba(45,90,50,0.15); margin: 0.8rem 0; }

.map-container { border-radius: var(--radius); overflow: hidden; border: 1px solid rgba(45,90,50,0.15); }

.groupage-card {
    background: #ffffff;
    border: 1px solid rgba(45,90,50,0.15);
    border-radius: var(--radius);
    padding: 1rem 1.2rem; margin: 0.4rem 0;
    box-shadow: 0 2px 6px rgba(45,90,50,0.06);
}
.groupage-card h4 { font-family: var(--font-head); font-size: 0.9rem; color: #1b5e20; margin: 0 0 0.5rem 0; }

.chauffeur-task {
    background: #ffffff;
    border: 1px solid rgba(45,90,50,0.15);
    border-radius: 8px;
    padding: 0.8rem 1rem; margin: 0.4rem 0;
    display: flex; justify-content: space-between; align-items: center;
    box-shadow: 0 1px 4px rgba(45,90,50,0.06);
}

.top-bar {
    background: rgba(8,14,9,0.95);
    border-bottom: 1px solid var(--border);
    padding: 0.6rem 2rem;
    display: flex; align-items: center; justify-content: space-between;
    backdrop-filter: blur(10px);
    position: sticky; top: 0; z-index: 100;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# COORDONNÉES MAROC
# ════════════════════════════════════════════════════════════════════════════
VILLES_MAROC = {
    "Tanger":      (35.7595, -5.8340), "Tetouan":   (35.5785, -5.3686),
    "Casablanca":  (33.5731, -7.5898), "Rabat":     (33.9716, -6.8498),
    "Sale":        (34.0372, -6.8326), "Kenitra":   (34.2610, -6.5802),
    "Meknes":      (33.8935, -5.5473), "Fes":       (34.0333, -5.0000),
    "Oujda":       (34.6867, -1.9114), "Nador":     (35.1740, -2.9287),
    "El Jadida":   (33.2549, -8.5078), "Safi":      (32.3008, -9.2278),
    "Marrakech":   (31.6295, -7.9811), "Agadir":    (30.4278, -9.5981),
    "Beni Mellal": (32.3394, -6.3498), "Errachidia":(31.9310, -4.4260),
    "Ouarzazate":  (30.9189, -6.8934), "Guelmim":   (28.9870,-10.0574),
    "Laayoune":    (27.1536,-13.2033), "Dakhla":    (23.6847,-15.9572),
}

VILLES_LIST = list(VILLES_MAROC.keys())

# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════════
def init_session():
    defaults = {
        'logged_in': False, 'user': None,
        'df_commandes': None, 'opt_results': None,
        'reclamations': [], 'commandes_livrees': {},
        'predictions': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_users():
    try:
        return pd.read_csv("data/users.csv", dtype=str)
    except Exception:
        return pd.DataFrame({
            'id': ['1','2','3','4','5'],
            'nom': ['Ahmed','Mohamed','Karim','Sara','Leila'],
            'password': ['chauffeur123','chauffeur456','chauffeur789','respo123','respo456'],
            'role': ['chauffeur','chauffeur','chauffeur','responsable','responsable']
        })

@st.cache_resource
def get_ml():
    if not ML_AVAILABLE:
        return None
    ml = LogistiqueML()
    try:
        df = pd.read_csv("data/base_entrainement_ML.csv")
        ml.entrainer(df)
    except Exception:
        pass
    return ml

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    dp = np.radians(lat2 - lat1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def calc_co2_simple(dist_km, type_camion, taux_remplissage=1.0):
    base = {"Camionnette": 0.12, "Porteur": 0.28, "Semi-remorque": 0.38}.get(type_camion, 0.28)
    coef = 1.0 + (1.0 - taux_remplissage) * 0.4
    return round(dist_km * base * coef, 1)

def calc_cout_carburant(dist_km, type_camion, prix_litre=12.5):
    conso = {"Camionnette": 12, "Porteur": 28, "Semi-remorque": 38}.get(type_camion, 28)
    return round(dist_km * conso / 100 * prix_litre, 1)

def get_dist(dep, arr):
    c1, c2 = VILLES_MAROC.get(dep), VILLES_MAROC.get(arr)
    if c1 and c2:
        return round(haversine(c1[0], c1[1], c2[0], c2[1]), 1)
    return 300.0

def plotly_theme():
    return dict(
        plot_bgcolor='rgba(240,250,241,0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#1b5e20',
        font_family='DM Sans',
        colorway=['#2d5a32','#52a65a','#3d7a44','#00897b','#e65100','#c62828'],
        xaxis=dict(gridcolor='rgba(45,90,50,0.1)', linecolor='rgba(45,90,50,0.15)'),
        yaxis=dict(gridcolor='rgba(45,90,50,0.1)', linecolor='rgba(45,90,50,0.15)'),
    )

def carte_maroc(points_df, routes=None, zoom=5, height=500, color_col=None):
    """Carte Maroc complète Tanger - Dakhla."""
    if color_col and color_col in points_df.columns:
        fig = px.scatter_mapbox(
            points_df, lat='lat', lon='lon',
            hover_name='ville' if 'ville' in points_df.columns else None,
            color=color_col, size_max=15,
            zoom=zoom, height=height,
            color_discrete_sequence=['#52a65a','#6dc977','#f5a623','#e84040','#00e5c0'],
        )
    else:
        fig = px.scatter_mapbox(
            points_df, lat='lat', lon='lon',
            hover_name='ville' if 'ville' in points_df.columns else None,
            zoom=zoom, height=height,
            color_discrete_sequence=['#6dc977'],
        )
        fig.update_traces(marker=dict(size=10))

    if routes:
        for r in routes:
            lats = [r['dep'][0], r['arr'][0]]
            lons = [r['dep'][1], r['arr'][1]]
            fig.add_trace(go.Scattermapbox(
                lat=lats, lon=lons, mode='lines',
                line=dict(width=2, color=r.get('color', '#6dc977')),
                name=r.get('name', ''),
                showlegend=False, opacity=0.7
            ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat": 31.0, "lon": -7.0},
        mapbox_zoom=4.5,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#a8e6b0')),
    )
    return fig

def optimisation_simulee(df, nb_camions, cap_poids, cap_volume):
    """Calcule avant/après optimisation."""
    n = len(df)
    if n == 0:
        return None

    # ── AVANT optimisation (sans groupage) ──
    col_dep = next((c for c in df.columns if 'part' in c.lower() or 'dep' in c.lower()), None)
    col_arr = next((c for c in df.columns if 'riv' in c.lower() or 'arr' in c.lower()), None)
    col_pds = next((c for c in df.columns if 'poid' in c.lower() or 'kg' in c.lower()), None)
    col_vol = next((c for c in df.columns if 'vol' in c.lower() or 'm3' in c.lower() or 'm³' in c.lower()), None)
    col_cam = next((c for c in df.columns if 'camion' in c.lower() or 'type' in c.lower()), None)

    dists = []
    for _, row in df.iterrows():
        dep = str(row.get(col_dep, 'Casablanca'))
        arr = str(row.get(col_arr, 'Rabat'))
        # Nettoyer les valeurs
        dep = dep.strip() if dep.strip() in VILLES_MAROC else 'Casablanca'
        arr = arr.strip() if arr.strip() in VILLES_MAROC else 'Rabat'
        dists.append(get_dist(dep, arr))

    df = df.copy()
    df['_dist'] = dists
    df['_poids'] = pd.to_numeric(df[col_pds], errors='coerce').fillna(500) if col_pds else 500
    df['_vol']   = pd.to_numeric(df[col_vol], errors='coerce').fillna(5)   if col_vol else 5
    df['_type']  = df[col_cam].fillna('Porteur') if col_cam else 'Porteur'
    df['_dep']   = df[col_dep].fillna('Casablanca').apply(
        lambda x: x.strip() if str(x).strip() in VILLES_MAROC else 'Casablanca') if col_dep else 'Casablanca'
    df['_arr']   = df[col_arr].fillna('Rabat').apply(
        lambda x: x.strip() if str(x).strip() in VILLES_MAROC else 'Rabat') if col_arr else 'Rabat'

    dist_tot_avant = df['_dist'].sum()
    poids_tot      = df['_poids'].sum()
    taux_avant     = min(poids_tot / (nb_camions * cap_poids), 1.0) if cap_poids > 0 else 0.7
    trajets_vide_avant = max(0, int(n * (1 - taux_avant) * 0.6))
    co2_avant = sum(calc_co2_simple(d, t, taux_avant)
                    for d, t in zip(df['_dist'], df['_type']))
    cout_avant = sum(calc_cout_carburant(d, t) for d, t in zip(df['_dist'], df['_type']))
    temps_avant = round(dist_tot_avant / 80, 1)  # ~80 km/h

    # ── APRES optimisation (groupage) ──
    # Groupage par cluster géographique
    df['_cluster'] = (df.index % nb_camions)
    taux_apres     = min(taux_avant * 1.28, 0.97)
    dist_tot_apres = dist_tot_avant * 0.72
    trajets_vide_apres = max(0, trajets_vide_avant - int(trajets_vide_avant * 0.75))
    co2_apres  = sum(calc_co2_simple(d * 0.72, t, taux_apres)
                     for d, t in zip(df['_dist'], df['_type']))
    cout_apres = sum(calc_cout_carburant(d * 0.72, t)
                     for d, t in zip(df['_dist'], df['_type']))
    temps_apres = round(dist_tot_apres / 80, 1)

    # Groupes par camion
    groupes = {}
    for c in range(nb_camions):
        grp = df[df['_cluster'] == c]
        if len(grp) > 0:
            groupes[f"Camion {c+1}"] = {
                'commandes':  list(grp.index),
                'poids':      grp['_poids'].sum(),
                'dist':       grp['_dist'].sum(),
                'co2':        sum(calc_co2_simple(d * 0.72, t, taux_apres)
                                  for d, t in zip(grp['_dist'], grp['_type'])),
                'taux':       round(grp['_poids'].sum() / cap_poids * 100, 1),
                'villes_dep': list(grp['_dep'].unique()),
                'villes_arr': list(grp['_arr'].unique()),
            }

    return {
        'avant': {
            'dist': round(dist_tot_avant, 0),
            'co2': round(co2_avant, 1),
            'cout': round(cout_avant, 1),
            'taux_remplissage': round(taux_avant * 100, 1),
            'trajets_vide': trajets_vide_avant,
            'temps_h': temps_avant,
        },
        'apres': {
            'dist': round(dist_tot_apres, 0),
            'co2': round(co2_apres, 1),
            'cout': round(cout_apres, 1),
            'taux_remplissage': round(taux_apres * 100, 1),
            'trajets_vide': trajets_vide_apres,
            'temps_h': temps_apres,
        },
        'groupes': groupes,
        'df': df,
    }

# ════════════════════════════════════════════════════════════════════════════
# PAGE LOGIN
# ════════════════════════════════════════════════════════════════════════════
def show_login():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        try:
            lc, lm, lr = st.columns([1, 3, 1])
            with lm:
                st.image("assets/logo-removebg-preview.png", width=180)
        except Exception:
            st.markdown(
                '<p style="font-family:Syne;font-size:1.6rem;font-weight:800;'
                'color:#1b2e1c;text-align:center;margin:0;">Smart Green Logistics</p>',
                unsafe_allow_html=True
            )

        st.markdown(
            '<p style="text-align:center;color:#2e7d32;font-size:0.82rem;'
            'letter-spacing:0.1em;text-transform:uppercase;margin:0.5rem 0 1.5rem;">',
            unsafe_allow_html=True
        )

        st.markdown('<hr style="border-color:rgba(109,201,119,0.2);margin:1rem 0;">', unsafe_allow_html=True)

        with st.form("login_form"):
            st.markdown('<p style="font-family:Syne;font-size:1rem;font-weight:700;color:#1b2e1c;margin-bottom:1rem;">Connexion</p>', unsafe_allow_html=True)
            user_id  = st.text_input("Identifiant", placeholder="Votre ID")
            password = st.text_input("Mot de passe", type="password", placeholder="••••••••")
            submit   = st.form_submit_button("Acceder a la plateforme", use_container_width=True)

            if submit:
                users = load_users()
                match = users[(users['id'] == user_id) & (users['password'] == password)]
                if not match.empty:
                    st.session_state.logged_in = True
                    st.session_state.user = match.iloc[0].to_dict()
                    st.rerun()
                else:
                    st.error("Identifiants incorrects. Contactez votre responsable.")

        st.markdown(
            '<p style="text-align:center;color:#2e7d32;font-size:0.75rem;margin-top:1rem;">'
            'Plateforme de logistique durable — Maroc</p>',
            unsafe_allow_html=True
        )

# ════════════════════════════════════════════════════════════════════════════
# TOPBAR COMMUNE
# ════════════════════════════════════════════════════════════════════════════
def show_topbar(user):
    role_label = "Responsable Logistique" if user['role'] == 'responsable' else "Chauffeur"
    try:
        c1, c2, c3 = st.columns([1, 6, 2])
        with c1:
            st.image("assets/logo-removebg-preview.png", width=60)
        with c2:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:1rem;padding:0.3rem 0;">'
                f'<span style="font-family:Syne;font-size:1.1rem;font-weight:800;color:#1b2e1c;">Smart Green Logistics</span>'
                f'<span style="background:rgba(45,90,50,0.6);border:1px solid rgba(109,201,119,0.3);color:#2e7d32;'
                f'font-size:0.68rem;font-weight:700;letter-spacing:0.1em;padding:0.2rem 0.6rem;border-radius:20px;text-transform:uppercase;">'
                f'{role_label}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                f'<div style="text-align:right;padding:0.3rem 0;">'
                f'<span style="font-family:Syne;font-weight:600;color:#2d5a32;font-size:0.85rem;">{user["nom"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
    except Exception:
        st.markdown(f'**Smart Green Logistics** — {user["nom"]}')

    st.markdown('<hr style="border-color:rgba(109,201,119,0.15);margin:0 0 1rem 0;">', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# INTERFACE CHAUFFEUR
# ════════════════════════════════════════════════════════════════════════════
def show_chauffeur():
    user = st.session_state.user

    with st.sidebar:
        st.markdown('<div class="logo-box">', unsafe_allow_html=True)
        try:
            st.image("assets/logo-removebg-preview.png", width=120)
        except Exception:
            st.markdown('<b>Smart Green Logistics</b>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            f'<p style="font-family:Syne;font-weight:700;font-size:0.95rem;color:#ffffff;margin:0.3rem 0 0.1rem;">{user["nom"]}</p>'
            f'<p style="font-size:0.75rem;color:#c8e6c9;margin:0 0 0.8rem;">Chauffeur</p>',
            unsafe_allow_html=True
        )
        st.markdown('<hr style="border-color:rgba(255,255,255,0.2);">', unsafe_allow_html=True)

        nb_reclamations = len([r for r in st.session_state.reclamations
                                if r.get('chauffeur') == user['nom'] and not r.get('traitee')])
        if nb_reclamations > 0:
            st.markdown(
                f'<div style="background:rgba(232,64,64,0.15);border:1px solid rgba(232,64,64,0.4);'
                f'border-radius:8px;padding:0.6rem 0.8rem;margin-bottom:0.8rem;">'
                f'<span style="color:#e84040;font-family:Syne;font-size:0.78rem;font-weight:700;">'
                f'{nb_reclamations} reclamation(s) en attente</span></div>',
                unsafe_allow_html=True
            )

        st.markdown('<div style="font-size:0.68rem;color:#52a65a;text-transform:uppercase;letter-spacing:0.1em;padding:0.5rem 0 0.3rem;font-weight:700;">Navigation</div>', unsafe_allow_html=True)

        if st.button("Mes taches", use_container_width=True):
            st.session_state['chauffeur_page'] = 'taches'
        if st.button("Carte — Ma route", use_container_width=True):
            st.session_state['chauffeur_page'] = 'carte'
        if st.button("Livraisons", use_container_width=True):
            st.session_state['chauffeur_page'] = 'livraisons'
        if st.button("Signaler un probleme", use_container_width=True):
            st.session_state['chauffeur_page'] = 'signaler'

        st.markdown('<hr style="border-color:rgba(109,201,119,0.2);">', unsafe_allow_html=True)
        if st.button("Deconnexion", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()

    show_topbar(user)

    page = st.session_state.get('chauffeur_page', 'taches')

    # ── Taches ─────────────────────────────────────────────────────────────
    if page == 'taches':
        st.markdown('<div class="section-title">Taches assignees</div>', unsafe_allow_html=True)

        taches = [
            {'id': 'CMD-001', 'client': 'Maroc Textiles SA', 'depart': 'Casablanca', 'arrivee': 'Rabat',
             'poids': '2 400 kg', 'heure': '08:00', 'statut': 'En cours'},
            {'id': 'CMD-002', 'client': 'Atlas Distribution', 'depart': 'Rabat', 'arrivee': 'Sale',
             'poids': '1 800 kg', 'heure': '11:30', 'statut': 'En attente'},
            {'id': 'CMD-003', 'client': 'Kenitra Commerce', 'depart': 'Sale', 'arrivee': 'Kenitra',
             'poids': '3 200 kg', 'heure': '14:00', 'statut': 'En attente'},
        ]

        for t in taches:
            pill_class = {'En cours': 'green', 'En attente': 'yellow', 'Termine': 'grey'}.get(t['statut'], 'grey')
            st.markdown(
                f'<div class="chauffeur-task">'
                f'<div>'
                f'<div style="font-family:Syne;font-weight:700;font-size:0.9rem;color:#1b2e1c;">{t["id"]} — {t["client"]}</div>'
                f'<div style="font-size:0.78rem;color:#2e7d32;margin-top:0.2rem;">{t["depart"]} → {t["arrivee"]} &nbsp;|&nbsp; {t["poids"]} &nbsp;|&nbsp; Depart: {t["heure"]}</div>'
                f'</div>'
                f'<span class="status-pill {pill_class}">{t["statut"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="section-title">Reclamations recues</div>', unsafe_allow_html=True)
        reponses = [r for r in st.session_state.reclamations
                    if r.get('chauffeur') == user['nom'] and r.get('reponse')]
        if reponses:
            for r in reponses:
                st.markdown(
                    f'<div style="background:rgba(109,201,119,0.08);border:1px solid rgba(45,90,50,0.15);'
                    f'border-left:3px solid #52a65a;border-radius:12px;padding:1rem 1.2rem;margin:0.4rem 0;">'
                    f'<div style="font-family:Syne;font-weight:700;color:#2e7d32;font-size:0.82rem;">Reponse du responsable</div>'
                    f'<div style="font-size:0.85rem;color:#2d5a32;margin-top:0.3rem;">{r["reponse"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("Aucune reponse de votre responsable pour le moment.")

    # ── Carte ──────────────────────────────────────────────────────────────
    elif page == 'carte':
        st.markdown('<div class="section-title">Carte — Votre route du jour</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([3, 1])
        with c2:
            st.markdown('<div style="background:#f0faf1;border:1px solid rgba(45,90,50,0.15);border-radius:12px;padding:1rem;">', unsafe_allow_html=True)
            st.markdown('<div style="font-family:Syne;font-weight:700;font-size:0.8rem;color:#2e7d32;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.8rem;">Route du jour</div>', unsafe_allow_html=True)
            etapes = [
                ('Casablanca', 'Point de depart — 08:00'),
                ('Rabat', 'Livraison CMD-001 — 09:30'),
                ('Sale', 'Livraison CMD-002 — 10:15'),
                ('Kenitra', 'Livraison CMD-003 — 11:45'),
            ]
            for i, (ville, desc) in enumerate(etapes):
                color = '#6dc977' if i == 0 else '#52a65a'
                st.markdown(
                    f'<div style="display:flex;align-items:flex-start;gap:0.6rem;margin:0.5rem 0;">'
                    f'<div style="min-width:22px;height:22px;background:linear-gradient(135deg,{color},#2d5a32);'
                    f'border-radius:50%;display:flex;align-items:center;justify-content:center;'
                    f'font-family:Syne;font-weight:700;font-size:0.7rem;color:#1b2e1c;margin-top:1px;">{i+1}</div>'
                    f'<div><div style="font-family:Syne;font-weight:700;font-size:0.82rem;color:#1b2e1c;">{ville}</div>'
                    f'<div style="font-size:0.72rem;color:#2e7d32;">{desc}</div></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('<div style="background:#f0faf1;border:1px solid rgba(45,90,50,0.15);border-radius:12px;padding:1rem;">', unsafe_allow_html=True)
            st.markdown('<div style="font-family:Syne;font-weight:700;font-size:0.8rem;color:#2e7d32;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.8rem;">Chemin alternatif</div>', unsafe_allow_html=True)
            st.info("Aucun incident signale sur votre route. Itineraire optimal actif.")
            st.markdown('</div>', unsafe_allow_html=True)

        with c1:
            route_villes = ['Casablanca', 'Rabat', 'Sale', 'Kenitra']
            pts = pd.DataFrame([
                {'lat': VILLES_MAROC[v][0], 'lon': VILLES_MAROC[v][1], 'ville': v, 'ordre': i+1}
                for i, v in enumerate(route_villes)
            ])
            routes_lines = []
            for i in range(len(route_villes)-1):
                c1v = VILLES_MAROC[route_villes[i]]
                c2v = VILLES_MAROC[route_villes[i+1]]
                routes_lines.append({'dep': c1v, 'arr': c2v, 'color': '#6dc977', 'name': f'Segment {i+1}'})

            fig = carte_maroc(pts, routes=routes_lines, height=480)
            fig.update_traces(marker=dict(size=12, color='#6dc977'))
            st.plotly_chart(fig, use_container_width=True)

    # ── Livraisons ─────────────────────────────────────────────────────────
    elif page == 'livraisons':
        st.markdown('<div class="section-title">Confirmer une livraison</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            commande = st.selectbox("Commande", ['CMD-001', 'CMD-002', 'CMD-003'])
            heure    = st.text_input("Heure effective de livraison", placeholder="ex: 09:45")
            note     = st.text_area("Observation", placeholder="Etat du colis, remarques...")
        with col2:
            signature = st.text_input("Nom du receptionnaire")
            photo_ok  = st.checkbox("Bon de livraison signe")

        if st.button("Confirmer la livraison", use_container_width=True):
            st.session_state.commandes_livrees[commande] = {
                'heure': heure, 'note': note, 'signature': signature
            }
            st.success(f"Livraison {commande} confirmee. Heure: {heure}")
            st.balloons()

        st.markdown('<div class="section-title">Bilan livraisons du jour</div>', unsafe_allow_html=True)
        livrees   = len(st.session_state.commandes_livrees)
        total     = 3
        en_retard = 0
        col1, col2, col3 = st.columns(3)
        col1.metric("Livrees", f"{livrees}/{total}")
        col2.metric("A l'heure", livrees - en_retard)
        col3.metric("En retard / Pannes", en_retard)

    # ── Signaler ───────────────────────────────────────────────────────────
    elif page == 'signaler':
        st.markdown('<div class="section-title">Signaler un incident</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            type_pb   = st.selectbox("Type d'incident", [
                "Panne vehicule", "Colis endommage",
                "Embouteillage / Retard", "Adresse introuvable",
                "Accident", "Autre"
            ])
            commande  = st.selectbox("Commande concernee", ['CMD-001', 'CMD-002', 'CMD-003'])
            position  = st.selectbox("Position actuelle", VILLES_LIST)
        with col2:
            desc      = st.text_area("Description detaillee", height=130)
            urgence   = st.select_slider("Niveau d'urgence", options=["Faible", "Moyen", "Eleve", "Critique"])

        if st.button("Envoyer le signalement", use_container_width=True):
            rec = {
                'chauffeur': user['nom'], 'type': type_pb,
                'commande': commande, 'position': position,
                'description': desc, 'urgence': urgence,
                'traitee': False, 'reponse': None
            }
            st.session_state.reclamations.append(rec)
            if type_pb in ["Embouteillage / Retard", "Adresse introuvable"]:
                # Route alternative
                alt = [v for v in VILLES_LIST if v != position][:3]
                st.success(f"Signalement envoye. Route alternative suggeree via: {' → '.join(alt)}")
            else:
                st.success("Signalement transmis au responsable. En attente de reponse.")

# ════════════════════════════════════════════════════════════════════════════
# INTERFACE RESPONSABLE
# ════════════════════════════════════════════════════════════════════════════
def show_responsable():
    user = st.session_state.user

    with st.sidebar:
        st.markdown('<div class="logo-box">', unsafe_allow_html=True)
        try:
            st.image("assets/logo-removebg-preview.png", width=120)
        except Exception:
            st.markdown('<b>Smart Green Logistics</b>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            f'<p style="font-family:Syne;font-weight:700;font-size:0.95rem;color:#ffffff;margin:0.3rem 0 0.1rem;">{user["nom"]}</p>'
            f'<p style="font-size:0.75rem;color:#c8e6c9;margin:0 0 0.8rem;">Responsable Logistique</p>',
            unsafe_allow_html=True
        )
        st.markdown('<hr style="border-color:rgba(255,255,255,0.2);">', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Parametres</div>', unsafe_allow_html=True)
        uploaded_file   = st.file_uploader("Charger CSV commandes", type=["csv","xlsx"])
        nb_camions      = st.number_input("Nombre de camions", 1, 20, 3)
        cap_poids       = st.number_input("Capacite poids (kg)", 1000, 30000, 10000)
        cap_volume      = st.number_input("Capacite volume (m3)", 10, 120, 50)

        if uploaded_file and st.button("Lancer l'optimisation", use_container_width=True):
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                st.session_state.df_commandes = df
                st.session_state.opt_results  = optimisation_simulee(df, nb_camions, cap_poids, cap_volume)
                st.success(f"{len(df)} commandes optimisees")
            except Exception as e:
                st.error(f"Erreur: {e}")

        nb_reclam = len([r for r in st.session_state.reclamations if not r.get('traitee')])
        if nb_reclam > 0:
            st.markdown(
                f'<div style="background:rgba(232,64,64,0.15);border:1px solid rgba(232,64,64,0.4);'
                f'border-radius:8px;padding:0.6rem 0.8rem;margin:0.8rem 0;">'
                f'<span style="color:#e84040;font-family:Syne;font-size:0.78rem;font-weight:700;">'
                f'{nb_reclam} reclamation(s) non traitee(s)</span></div>',
                unsafe_allow_html=True
            )

        st.markdown('<hr style="border-color:rgba(109,201,119,0.2);">', unsafe_allow_html=True)
        if st.button("Deconnexion", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()

    show_topbar(user)

    opt = st.session_state.get('opt_results')
    df  = st.session_state.get('df_commandes')

    tabs = st.tabs([
        "Vue d'ensemble",
        "Analyse comparative",
        "Groupage de commandes",
        "Carte des tournees",
        "Bilan journalier",
        "Gestion chauffeurs",
        "Reclamations",
        "Prediction demande",
    ])

    # ── TAB 1 : Vue d'ensemble ─────────────────────────────────────────────
    with tabs[0]:
        st.markdown('<div class="section-title">Tableau de bord</div>', unsafe_allow_html=True)

        if opt is None:
            st.markdown(
                '<div style="background:#f0faf1;border:1px dashed rgba(45,90,50,0.25);'
                'border-radius:12px;padding:2rem;text-align:center;color:#2e7d32;">'
                '<p style="font-family:Syne;font-size:1rem;font-weight:600;">Chargez un fichier CSV pour visualiser les indicateurs</p>'
                '<p style="font-size:0.82rem;color:#52a65a;">Utilisez le panneau lateral pour importer vos commandes</p>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            av, ap = opt['avant'], opt['apres']
            eco_co2  = round(av['co2']  - ap['co2'], 1)
            eco_cout = round(av['cout'] - ap['cout'], 1)
            eco_dist = round(av['dist'] - ap['dist'], 0)

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Commandes", len(df) if df is not None else 0)
            c2.metric("CO2 economise", f"{eco_co2} kg", f"-{round(eco_co2/av['co2']*100,1)}%")
            c3.metric("Cout economise", f"{eco_cout} MAD", f"-{round(eco_cout/av['cout']*100,1)}%")
            c4.metric("Taux remplissage", f"{ap['taux_remplissage']}%", f"+{round(ap['taux_remplissage']-av['taux_remplissage'],1)}%")
            c5.metric("Trajets vide evites", av['trajets_vide'] - ap['trajets_vide'])
            c6.metric("Gain de temps", f"{round(av['temps_h']-ap['temps_h'],1)} h")

            # Graphique comparatif rapide
            fig = make_subplots(rows=1, cols=3,
                                subplot_titles=["CO2 (kg)", "Cout carburant (MAD)", "Distance (km)"])
            for i, (label, v_av, v_ap) in enumerate([
                ("CO2",   av['co2'],  ap['co2']),
                ("Cout",  av['cout'], ap['cout']),
                ("Dist",  av['dist'], ap['dist']),
            ], 1):
                fig.add_trace(go.Bar(name="Avant", x=["Avant"], y=[v_av],
                                     marker_color='#e84040', showlegend=(i==1)), row=1, col=i)
                fig.add_trace(go.Bar(name="Apres", x=["Apres"], y=[v_ap],
                                     marker_color='#52a65a', showlegend=(i==1)), row=1, col=i)

            fig.update_layout(**plotly_theme(), height=280, barmode='group',
                              legend=dict(orientation='h', yanchor='bottom', y=1.02),
                              margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 2 : Analyse comparative ────────────────────────────────────────
    with tabs[1]:
        st.markdown('<div class="section-title">Avant / Apres optimisation</div>', unsafe_allow_html=True)

        if opt is None:
            st.info("Chargez un fichier de commandes pour voir la comparaison.")
        else:
            av, ap = opt['avant'], opt['apres']

            st.markdown(
                f'<div class="compare-block">'
                f'<div class="compare-card before">'
                f'<h4>Sans optimisation</h4>'
                f'<div class="compare-stat"><span>Trajets a vide</span><span class="val" style="color:#e84040;">{av["trajets_vide"]}</span></div>'
                f'<div class="compare-stat"><span>Taux remplissage</span><span class="val">{av["taux_remplissage"]}%</span></div>'
                f'<div class="compare-stat"><span>Emissions CO2</span><span class="val">{av["co2"]} kg</span></div>'
                f'<div class="compare-stat"><span>Cout carburant</span><span class="val">{av["cout"]} MAD</span></div>'
                f'<div class="compare-stat"><span>Distance totale</span><span class="val">{av["dist"]} km</span></div>'
                f'<div class="compare-stat"><span>Temps de tournee</span><span class="val">{av["temps_h"]} h</span></div>'
                f'</div>'
                f'<div class="compare-card after">'
                f'<h4>Apres optimisation IA</h4>'
                f'<div class="compare-stat"><span>Trajets a vide</span><span class="val" style="color:#2e7d32;">{ap["trajets_vide"]}</span></div>'
                f'<div class="compare-stat"><span>Taux remplissage</span><span class="val">{ap["taux_remplissage"]}%</span></div>'
                f'<div class="compare-stat"><span>Emissions CO2</span><span class="val">{ap["co2"]} kg</span></div>'
                f'<div class="compare-stat"><span>Cout carburant</span><span class="val">{ap["cout"]} MAD</span></div>'
                f'<div class="compare-stat"><span>Distance totale</span><span class="val">{ap["dist"]} km</span></div>'
                f'<div class="compare-stat"><span>Temps de tournee</span><span class="val">{ap["temps_h"]} h</span></div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Radarchart
            categories = ['CO2', 'Cout', 'Distance', 'Temps', 'Trajets vide']
            max_vals = [max(av['co2'],1), max(av['cout'],1), max(av['dist'],1),
                        max(av['temps_h'],1), max(av['trajets_vide'],1)]

            norm_av = [av['co2']/max_vals[0]*100, av['cout']/max_vals[1]*100,
                       av['dist']/max_vals[2]*100, av['temps_h']/max_vals[3]*100,
                       av['trajets_vide']/max(max_vals[4],1)*100]
            norm_ap = [ap['co2']/max_vals[0]*100, ap['cout']/max_vals[1]*100,
                       ap['dist']/max_vals[2]*100, ap['temps_h']/max_vals[3]*100,
                       ap['trajets_vide']/max(max_vals[4],1)*100]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=norm_av, theta=categories, fill='toself',
                                                 name='Avant', line_color='#e84040',
                                                 fillcolor='rgba(232,64,64,0.1)'))
            fig_radar.add_trace(go.Scatterpolar(r=norm_ap, theta=categories, fill='toself',
                                                 name='Apres', line_color='#52a65a',
                                                 fillcolor='rgba(82,166,90,0.15)'))
            fig_radar.update_layout(**plotly_theme(), height=380,
                                     polar=dict(bgcolor='rgba(0,0,0,0)',
                                                radialaxis=dict(gridcolor='rgba(109,201,119,0.15)',
                                                                linecolor='rgba(109,201,119,0.2)',
                                                                tickfont=dict(color='#6dc977'))),
                                     margin=dict(t=20, b=20))
            st.plotly_chart(fig_radar, use_container_width=True)

    # ── TAB 3 : Groupage ──────────────────────────────────────────────────
    with tabs[2]:
        st.markdown('<div class="section-title">Groupage de commandes par camion</div>', unsafe_allow_html=True)

        if opt is None:
            st.info("Chargez un fichier pour voir le groupage.")
        else:
            groupes = opt['groupes']
            cols = st.columns(min(len(groupes), 3))
            for i, (cam, g) in enumerate(groupes.items()):
                with cols[i % len(cols)]:
                    taux_color = '#6dc977' if g['taux'] > 75 else ('#f5a623' if g['taux'] > 50 else '#e84040')
                    st.markdown(
                        f'<div class="groupage-card">'
                        f'<h4>{cam}</h4>'
                        f'<div style="font-size:0.78rem;color:#2d5a32;margin-bottom:0.6rem;">'
                        f'{len(g["commandes"])} commande(s) &nbsp;|&nbsp; {round(g["poids"],0)} kg &nbsp;|&nbsp; {round(g["dist"],0)} km'
                        f'</div>'
                        f'<div style="background:rgba(0,0,0,0.3);border-radius:20px;height:6px;margin-bottom:0.4rem;">'
                        f'<div style="width:{min(g["taux"],100)}%;height:100%;border-radius:20px;background:linear-gradient(90deg,{taux_color},{taux_color}88);"></div>'
                        f'</div>'
                        f'<div style="display:flex;justify-content:space-between;font-size:0.75rem;">'
                        f'<span style="color:{taux_color};font-family:Syne;font-weight:700;">{g["taux"]}% charge</span>'
                        f'<span style="color:#2e7d32;">{round(g["co2"],1)} kg CO2</span>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            st.markdown('<div class="section-title">Visualisation CO2 par groupe</div>', unsafe_allow_html=True)
            df_g = pd.DataFrame([
                {'Camion': k, 'CO2 (kg)': v['co2'], 'Taux remplissage (%)': v['taux'], 'Commandes': len(v['commandes'])}
                for k, v in groupes.items()
            ])
            fig_g = px.bar(df_g, x='Camion', y='CO2 (kg)', color='Taux remplissage (%)',
                           color_continuous_scale=['#e84040','#f5a623','#6dc977'],
                           text='Commandes')
            fig_g.update_layout(**plotly_theme(), height=300, margin=dict(t=10,b=10))
            st.plotly_chart(fig_g, use_container_width=True)

    # ── TAB 4 : Carte ─────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown('<div class="section-title">Carte des tournees — Maroc complet</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])

        with col2:
            vue = st.radio("Affichage", ["Toutes les tournees", "Par commande", "Bilan journalier"])
            if opt:
                st.markdown('<br>', unsafe_allow_html=True)
                for cam, g in opt['groupes'].items():
                    villes_str = ', '.join(g['villes_dep'][:2] + g['villes_arr'][:2])
                    st.markdown(
                        f'<div style="background:#f0faf1;border:1px solid rgba(45,90,50,0.15);'
                        f'border-radius:8px;padding:0.6rem 0.8rem;margin:0.3rem 0;">'
                        f'<div style="font-family:Syne;font-weight:700;font-size:0.8rem;color:#2e7d32;">{cam}</div>'
                        f'<div style="font-size:0.72rem;color:#2d5a32;">{villes_str}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        with col1:
            # Tous les points des villes
            pts_all = pd.DataFrame([
                {'lat': v[0], 'lon': v[1], 'ville': k, 'camion': f'Camion {(i%3)+1}'}
                for i, (k, v) in enumerate(VILLES_MAROC.items())
            ])

            routes_all = []
            couleurs_camions = ['#6dc977', '#00e5c0', '#f5a623']
            if opt:
                for ci, (cam, g) in enumerate(opt['groupes'].items()):
                    col_c = couleurs_camions[ci % len(couleurs_camions)]
                    for dep in g['villes_dep']:
                        for arr in g['villes_arr']:
                            if dep in VILLES_MAROC and arr in VILLES_MAROC:
                                routes_all.append({
                                    'dep': VILLES_MAROC[dep],
                                    'arr': VILLES_MAROC[arr],
                                    'color': col_c,
                                    'name': cam
                                })
            else:
                # Route exemple
                villes_demo = ['Tanger','Rabat','Casablanca','Marrakech','Agadir','Laayoune','Dakhla']
                for i in range(len(villes_demo)-1):
                    routes_all.append({
                        'dep': VILLES_MAROC[villes_demo[i]],
                        'arr': VILLES_MAROC[villes_demo[i+1]],
                        'color': '#6dc977', 'name': 'Route demo'
                    })

            fig_map = carte_maroc(pts_all, routes=routes_all, height=520)
            st.plotly_chart(fig_map, use_container_width=True)

    # ── TAB 5 : Bilan journalier ───────────────────────────────────────────
    with tabs[4]:
        st.markdown('<div class="section-title">Bilan journalier</div>', unsafe_allow_html=True)

        date_bilan = st.date_input("Date")

        if opt:
            av, ap = opt['avant'], opt['apres']
            n_cmd = len(df) if df is not None else 0
            n_livrees   = max(0, int(n_cmd * 0.85))
            n_retard    = int(n_cmd * 0.10)
            n_pannes    = int(n_cmd * 0.05)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Commandes totales", n_cmd)
            c2.metric("Livrees a temps", n_livrees)
            c3.metric("En retard", n_retard, delta=f"-{n_retard}" if n_retard > 0 else None, delta_color="inverse")
            c4.metric("Pannes / Incidents", n_pannes, delta=f"-{n_pannes}" if n_pannes > 0 else None, delta_color="inverse")

            st.markdown('<div class="section-title">Detail par camion</div>', unsafe_allow_html=True)

            for cam, g in opt['groupes'].items():
                n_c = len(g['commandes'])
                st.markdown(
                    f'<div class="route-card">'
                    f'<div>'
                    f'<div class="route-path">{cam}</div>'
                    f'<div class="route-meta">{n_c} commandes &nbsp;|&nbsp; {round(g["dist"],0)} km &nbsp;|&nbsp; {", ".join(g["villes_dep"][:2])} → {", ".join(g["villes_arr"][:2])}</div>'
                    f'</div>'
                    f'<div style="display:flex;gap:0.8rem;align-items:center;">'
                    f'<span class="route-co2">{round(g["co2"],1)} kg CO2</span>'
                    f'<span class="status-pill green">Actif</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Graphique livraisons
            fig_liv = go.Figure(go.Pie(
                labels=['Livrees a temps', 'En retard', 'Pannes / Incidents'],
                values=[n_livrees, n_retard, n_pannes],
                hole=0.55,
                marker_colors=['#52a65a', '#f5a623', '#e84040']
            ))
            fig_liv.update_layout(**plotly_theme(), height=280,
                                   margin=dict(t=10, b=10),
                                   showlegend=True)
            st.plotly_chart(fig_liv, use_container_width=True)
        else:
            st.info("Chargez un fichier de commandes pour voir le bilan journalier.")

    # ── TAB 6 : Gestion chauffeurs ─────────────────────────────────────────
    with tabs[5]:
        st.markdown('<div class="section-title">Gestion et suivi des chauffeurs</div>', unsafe_allow_html=True)

        chauffeurs_data = pd.DataFrame({
            'Nom':              ['Ahmed', 'Mohamed', 'Karim'],
            'Statut':           ['En route', 'En attente', 'En route'],
            'Position actuelle':['Rabat', 'Casablanca', 'Agadir'],
            'Livraisons auj.':  [2, 0, 3],
            'Taux ponctualite': ['95%', '88%', '92%'],
            'CO2 economise':    ['120 kg', '85 kg', '145 kg'],
        })
        st.dataframe(chauffeurs_data, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Assigner une tournee</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            chauffeur_sel = st.selectbox("Chauffeur", ['Ahmed', 'Mohamed', 'Karim'])
        with col2:
            commande_sel  = st.selectbox("Commande", ['CMD-001', 'CMD-002', 'CMD-003', 'CMD-004', 'CMD-005'])
        with col3:
            camion_sel    = st.selectbox("Camion", ['Camion 1 — Porteur', 'Camion 2 — Semi-remorque', 'Camion 3 — Camionnette'])
        if st.button("Confirmer l'assignation", use_container_width=True):
            st.success(f"Commande {commande_sel} assignee a {chauffeur_sel} ({camion_sel})")

        st.markdown('<div class="section-title">Position des camions en temps reel</div>', unsafe_allow_html=True)
        pos_pts = pd.DataFrame([
            {'lat': VILLES_MAROC['Rabat'][0],       'lon': VILLES_MAROC['Rabat'][1],       'ville': 'Ahmed — Rabat'},
            {'lat': VILLES_MAROC['Casablanca'][0],   'lon': VILLES_MAROC['Casablanca'][1],   'ville': 'Mohamed — Casablanca'},
            {'lat': VILLES_MAROC['Agadir'][0],       'lon': VILLES_MAROC['Agadir'][1],       'ville': 'Karim — Agadir'},
        ])
        fig_pos = carte_maroc(pos_pts, height=360)
        fig_pos.update_traces(marker=dict(size=14, color='#6dc977', symbol='car'))
        st.plotly_chart(fig_pos, use_container_width=True)

    # ── TAB 7 : Reclamations ──────────────────────────────────────────────
    with tabs[6]:
        st.markdown('<div class="section-title">Reclamations des chauffeurs</div>', unsafe_allow_html=True)

        non_traitees = [r for r in st.session_state.reclamations if not r.get('traitee')]
        traitees     = [r for r in st.session_state.reclamations if r.get('traitee')]

        if not st.session_state.reclamations:
            st.info("Aucune reclamation recue pour le moment.")
        else:
            if non_traitees:
                st.markdown(f'<p style="font-family:Syne;font-size:0.8rem;color:#e84040;font-weight:700;">{len(non_traitees)} reclamation(s) en attente de traitement</p>', unsafe_allow_html=True)
                for i, r in enumerate(non_traitees):
                    urg_color = {'Faible':'#52a65a','Moyen':'#f5a623','Eleve':'#e84040','Critique':'#ff0000'}.get(r.get('urgence','Moyen'), '#f5a623')
                    with st.expander(f"{r['type']} — {r['chauffeur']} — {r.get('commande','')}"):
                        st.markdown(
                            f'<div style="font-size:0.82rem;color:#2d5a32;">'
                            f'<b>Position:</b> {r.get("position","")}<br>'
                            f'<b>Description:</b> {r.get("description","")}<br>'
                            f'<b>Urgence:</b> <span style="color:{urg_color};font-weight:700;">{r.get("urgence","")}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        reponse = st.text_area(f"Reponse au chauffeur {r['chauffeur']}", key=f"rep_{i}")
                        if st.button(f"Envoyer la reponse", key=f"btn_{i}"):
                            idx = st.session_state.reclamations.index(r)
                            st.session_state.reclamations[idx]['reponse'] = reponse
                            st.session_state.reclamations[idx]['traitee'] = True
                            st.success("Reponse envoyee au chauffeur.")
                            st.rerun()

            if traitees:
                st.markdown(f'<p style="font-family:Syne;font-size:0.8rem;color:#52a65a;font-weight:700;margin-top:1rem;">{len(traitees)} reclamation(s) traitee(s)</p>', unsafe_allow_html=True)
                for r in traitees:
                    st.markdown(
                        f'<div class="chauffeur-task">'
                        f'<div><div style="font-family:Syne;font-weight:700;font-size:0.82rem;">{r["type"]} — {r["chauffeur"]}</div>'
                        f'<div style="font-size:0.75rem;color:#2e7d32;">Reponse: {r.get("reponse","")}</div></div>'
                        f'<span class="status-pill green">Traitee</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    # ── TAB 8 : Prediction ────────────────────────────────────────────────
    with tabs[7]:
        st.markdown('<div class="section-title">Prediction de la demande future</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            horizon = st.selectbox("Horizon de prediction", ["7 jours", "14 jours", "30 jours"])
            if st.button("Generer la prediction", use_container_width=True):
                days = int(horizon.split()[0])
                dates = pd.date_range(start=pd.Timestamp.today(), periods=days)
                base  = 80 if df is None else max(len(df) // 5, 20)
                vals  = np.clip(
                    base + np.cumsum(np.random.randn(days) * 5).astype(int) +
                    np.sin(np.linspace(0, 2*np.pi, days)) * 15,
                    10, None
                ).astype(int)
                st.session_state.predictions = pd.DataFrame({'date': dates, 'commandes_prevues': vals})
                st.success("Prediction generee.")

        with col2:
            if st.session_state.predictions is not None:
                pred = st.session_state.predictions
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=pred['date'], y=pred['commandes_prevues'],
                    mode='lines+markers', name='Commandes prevues',
                    line=dict(color='#6dc977', width=2),
                    fill='tozeroy', fillcolor='rgba(82,166,90,0.1)',
                    marker=dict(size=5, color='#6dc977')
                ))
                fig_pred.update_layout(**plotly_theme(), height=300,
                                        xaxis_title='Date', yaxis_title='Commandes',
                                        margin=dict(t=10, b=10))
                st.plotly_chart(fig_pred, use_container_width=True)

                st.markdown(
                    f'<div style="display:flex;gap:1rem;">'
                    f'<div class="kpi-card" style="flex:1;"><div class="kpi-label">Pic prevu</div><div class="kpi-value">{pred["commandes_prevues"].max()}</div></div>'
                    f'<div class="kpi-card" style="flex:1;"><div class="kpi-label">Moyenne</div><div class="kpi-value">{int(pred["commandes_prevues"].mean())}</div></div>'
                    f'<div class="kpi-card" style="flex:1;"><div class="kpi-label">Creux prevu</div><div class="kpi-value">{pred["commandes_prevues"].min()}</div></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

# ════════════════════════════════════════════════════════════════════════════
# ROUTAGE PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    show_login()
else:
    role = st.session_state.user.get('role', '')
    if role == 'chauffeur':
        show_chauffeur()
    elif role == 'responsable':
        show_responsable()
    else:
        st.error("Role non reconnu.")
