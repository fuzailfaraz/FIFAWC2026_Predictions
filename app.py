import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import pickle
import shap
import matplotlib.pyplot as plt
import importlib
import simulator
importlib.reload(simulator) # Force reload to bypass cache
from simulator import run_tournament_simulation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Import custom modules
from predictor import predict_match
from player_analysis import radar_chart, compute_strength_index

# Set page config
st.set_page_config(
    page_title="FIFA World Cup 2026 Analytics",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium FIFA WC 2026 Theme
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Bebas+Neue&display=swap');
        
        /* Vibrant, Exciting WC2026 Layered Background */
        [data-testid="stAppViewContainer"] {
            background-color: #020617 !important;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(0, 178, 169, 0.4) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(255, 79, 129, 0.4) 0%, transparent 40%),
                radial-gradient(circle at 50% 50%, rgba(255, 200, 87, 0.2) 0%, transparent 50%),
                linear-gradient(rgba(10, 25, 47, 0.85), rgba(10, 25, 47, 0.85)),
                url("https://images.unsplash.com/photo-1522778119026-d647f0596c20?q=80&w=2000&auto=format&fit=crop") !important;
            background-size: 100% 100%, 100% 100%, 100% 100%, cover, cover !important;
            background-position: center center !important;
            background-attachment: fixed !important;
            animation: pulse-bg 15s infinite alternate !important;
            color: #f8fafc;
            font-family: 'Outfit', sans-serif;
        }
        
        @keyframes pulse-bg {
            0% { background-size: 100% 100%, 100% 100%, 100% 100%, cover, cover; }
            100% { background-size: 120% 120%, 120% 120%, 120% 120%, cover, cover; }
        }

        /* MASSIVE Global Sizing */
        html, body, [class*="css"], [class*="st-"] {
            font-size: 1.8rem !important;
        }
        p, li {
            font-size: 1.6rem !important;
            line-height: 1.8 !important;
        }
        h1 { font-size: 4.5rem !important; }
        h2 { font-size: 3.5rem !important; }
        h3 { font-size: 2.8rem !important; }
        .stMarkdown {
            font-size: 1.8rem !important;
        }

        /* Mobile Responsive Sizing */
        @media (max-width: 768px) {
            html, body, [class*="css"], [class*="st-"], .stMarkdown {
                font-size: 1.0rem !important;
            }
            p, li { font-size: 1.0rem !important; line-height: 1.4 !important; }
            h1 { font-size: 2.2rem !important; }
            h2 { font-size: 1.8rem !important; }
            h3 { font-size: 1.4rem !important; }
            
            .prediction-hero h1 { font-size: 3rem !important; }
            .prediction-hero h2 { font-size: 1.8rem !important; }
            .prediction-hero p { font-size: 1.2rem !important; }
            .hero-container::before { font-size: 10rem !important; }
            
            /* Fix wrapping for VS span */
            .prediction-hero span { min-width: 80px !important; font-size: 1.2rem !important; white-space: normal !important; word-break: break-word; }
        }

        /* Hero Container */
        .hero-container {
            position: relative;
            text-align: center;
            padding: 4rem 2rem;
            margin-bottom: 3rem;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 30px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 20px 50px rgba(0,0,0,0.5), inset 0 0 20px rgba(0, 178, 169, 0.1);
            backdrop-filter: blur(20px);
            overflow: hidden;
        }
        .hero-container::before {
            content: "⚽";
            position: absolute;
            font-size: 20rem;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            opacity: 0.03;
            animation: float 10s infinite ease-in-out;
            pointer-events: none;
        }

        /* Glassmorphism Sidebar */
        [data-testid="stSidebar"] {
            background: rgba(10, 25, 47, 0.65) !important;
            backdrop-filter: blur(12px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(12px) saturate(180%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        [data-testid="stSidebar"] h1 {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 2.8rem !important;
            background: linear-gradient(90deg, #00B2A9, #FF4F81);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 2px;
            white-space: nowrap;
        }

        /* Headings */
        h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 1.5px; }
        
        /* Hero Title */
        .hero-title {
            font-size: 5.5rem !important;
            text-align: center;
            background: linear-gradient(90deg, #00B2A9, #FFC857, #FF4F81);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(0, 178, 169, 0.3);
            margin-bottom: 0;
            line-height: 1.1;
        }
        .hero-subtitle {
            font-family: 'Outfit', sans-serif;
            font-size: 1.8rem;
            text-align: center;
            color: #94a3b8;
            margin-top: 0.5rem;
            margin-bottom: 3rem;
            font-weight: 300;
        }

        /* Streamlit Native Selectbox to Glassmorphism */
        div[data-baseweb="select"] > div {
            background: rgba(255, 255, 255, 0.03) !important;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 178, 169, 0.3) !important;
            border-radius: 12px;
            color: white !important;
            height: 4rem;
            font-size: 1.4rem;
            transition: all 0.3s ease;
        }
        div[data-baseweb="select"] > div:hover {
            border-color: #00B2A9 !important;
            box-shadow: 0 0 15px rgba(0, 178, 169, 0.3);
        }
        .stSelectbox label {
            font-family: 'Outfit', sans-serif;
            font-size: 1.3rem !important;
            color: #e2e8f0 !important;
            font-weight: 600;
        }

        /* Premium Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #00B2A9 0%, #007A74 100%);
            color: white !important;
            border: none;
            border-radius: 12px;
            height: 4.5rem;
            font-family: 'Bebas Neue', sans-serif;
            font-size: 2.2rem !important;
            letter-spacing: 2px;
            box-shadow: 0 10px 20px -5px rgba(0, 178, 169, 0.4);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        .stButton>button::after {
            content: ''; position: absolute; top: 0; left: -100%; width: 50%; height: 100%;
            background: linear-gradient(to right, transparent, rgba(255,255,255,0.3), transparent);
            transform: skewX(-20deg); transition: 0.5s;
        }
        .stButton>button:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 15px 25px -5px rgba(0, 178, 169, 0.6);
        }
        .stButton>button:hover::after { left: 150%; }

        /* Metric Cards - Glassmorphism */
        div[data-testid="metric-container"] {
            background: rgba(10, 25, 47, 0.65);
            backdrop-filter: blur(12px) saturate(180%);
            -webkit-backdrop-filter: blur(12px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 2.5rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            text-align: center;
            transition: transform 0.3s ease, border-color 0.3s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-5px);
            border-color: #00B2A9;
            box-shadow: 0 8px 32px 0 rgba(0, 178, 169, 0.4);
        }
        div[data-testid="stMetricValue"] {
            font-size: 4rem !important;
            font-weight: 800;
            background: linear-gradient(90deg, #00B2A9, #39ff14);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 20px rgba(0, 178, 169, 0.5);
        }
        
        /* Dramatic Prediction Result Cards */
        .prediction-hero {
            background: rgba(15, 23, 42, 0.6);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(0, 178, 169, 0.4);
            border-radius: 24px;
            padding: 3rem;
            text-align: center;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5), inset 0 0 20px rgba(0, 178, 169, 0.1);
            margin: 2rem 0;
            animation: float 6s ease-in-out infinite;
        }
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        /* Progress Bars */
        .prob-bar-container {
            width: 100%; height: 12px; background: rgba(255,255,255,0.1);
            border-radius: 6px; margin: 10px 0; overflow: hidden;
        }
        .prob-bar-fill-home { background: #00B2A9; height: 100%; transition: width 1.5s cubic-bezier(0.2, 0.8, 0.2, 1); }
        .prob-bar-fill-away { background: #FF4F81; height: 100%; transition: width 1.5s cubic-bezier(0.2, 0.8, 0.2, 1); }
        .prob-bar-fill-draw { background: #FFC857; height: 100%; transition: width 1.5s cubic-bezier(0.2, 0.8, 0.2, 1); }

        /* Dataframes */
        .dataframe {
            background: rgba(10, 25, 47, 0.65) !important;
            backdrop-filter: blur(12px) saturate(180%);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
            color: #e2e8f0;
            font-size: 1.4rem;
        }
        .dataframe th {
            background: rgba(0, 178, 169, 0.25) !important;
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            color: #39ff14 !important;
            padding: 1rem;
        }
        .dataframe td {
            padding: 1rem;
        }
        [data-testid="stDataFrame"] > div {
            background: rgba(10, 25, 47, 0.65) !important;
            backdrop-filter: blur(12px) saturate(180%) !important;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        /* Streamlit Tabs */
        button[data-baseweb="tab"] {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 1.8rem !important;
            letter-spacing: 1px;
            color: #94a3b8 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #00B2A9 !important;
            border-bottom: 3px solid #00B2A9 !important;
        }
        
        /* Divider */
        hr { border-color: rgba(255,255,255,0.1) !important; }
    </style>
""", unsafe_allow_html=True)

# ── DATA AND MODEL LOADERS ────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load model with fallbacks if pickle not generated yet."""
    model_path = 'models/xgb_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        # Fallback: Train a quick XGBoost model on the fly to avoid blank screens
        df_train = pd.read_csv('data/wc_training_dataset_updated.csv')
        FEATURE_COLS = [
            'home_fifa_rank', 'home_fifa_points', 'home_wc_win_rate',
            'home_avg_goals_scored', 'home_avg_goals_conceded', 'home_goal_diff',
            'home_wc_matches', 'home_wc_appearances', 'home_wc_titles',
            'away_fifa_rank', 'away_fifa_points', 'away_wc_win_rate',
            'away_avg_goals_scored', 'away_avg_goals_conceded', 'away_goal_diff',
            'away_wc_matches', 'away_wc_appearances', 'away_wc_titles',
            'rank_diff', 'points_diff', 'points_ratio',
            'win_rate_diff', 'goal_diff_diff', 'title_diff', 'experience_diff'
        ]
        X = df_train[FEATURE_COLS]
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(df_train['result'])
        
        model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
        model.fit(X, y)
        return model

@st.cache_resource
def load_explainer():
    """Load or build SHAP explainer."""
    explainer_path = 'models/shap_explainer.pkl'
    if os.path.exists(explainer_path):
        with open(explainer_path, 'rb') as f:
            return pickle.load(f)
    else:
        model_obj = load_model()
        return shap.TreeExplainer(model_obj)

def load_master():
    df = pd.read_csv('data/wc2026_master_dataset_updated.csv')
    df = df.rename(columns={
        'home_goal_diff_per_match': 'home_goal_diff',
        'away_goal_diff_per_match': 'away_goal_diff',
        'home_wc_matches_played': 'home_wc_matches',
        'away_wc_matches_played': 'away_wc_matches',
        'home_wc_appearances_94_22': 'home_wc_appearances',
        'away_wc_appearances_94_22': 'away_wc_appearances',
        'home_wc_appearances_94_18': 'home_wc_appearances',
        'away_wc_appearances_94_18': 'away_wc_appearances'
    })
    return df

def load_historical_teams():
    return pd.read_csv('data/team_historical_features_updated.csv')

@st.cache_data
def load_clusters():
    """Load or run clustering on the fly."""
    cluster_file = 'data/team_clusters.csv'
    _force_cache_clear = True # Added to force Streamlit cache invalidation
    if os.path.exists(cluster_file):
        return pd.read_csv(cluster_file)
    else:
        df_team = load_historical_teams()
        df_cluster = df_team[df_team['wc_matches_played'] >= 3].copy()
        CLUSTER_FEATURES = [
            'wc_win_rate', 'avg_goals_scored', 'avg_goals_conceded',
            'goal_diff_per_match', 'wc_appearances_94_22', 'wc_titles'
        ]
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        
        X = df_cluster[CLUSTER_FEATURES].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        df_cluster['pca1'] = X_pca[:, 0]
        df_cluster['pca2'] = X_pca[:, 1]
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df_cluster['cluster'] = kmeans.fit_predict(X_scaled)
        
        centroid_means = df_cluster.groupby('cluster')['wc_win_rate'].mean().sort_values(ascending=False)
        sorted_clusters = centroid_means.index.tolist()
        cluster_names = {}
        for rank, cluster_id in enumerate(sorted_clusters):
            if rank == 0:
                cluster_names[cluster_id] = 'Elite Powers'
            elif rank == 1:
                cluster_names[cluster_id] = 'Solid Contenders'
            elif rank == 2:
                cluster_names[cluster_id] = 'Attack-Heavy'
            else:
                cluster_names[cluster_id] = 'Defensive & Tight'
                
        df_cluster['cluster_name'] = df_cluster['cluster'].map(cluster_names)
        return df_cluster

# Load datasets and model
df_master = load_master()
df_team = load_historical_teams()
model = load_model()
explainer = load_explainer()
df_cluster = load_clusters()

FEATURE_COLS = [
    'home_fifa_rank', 'home_fifa_points', 'home_wc_win_rate',
    'home_avg_goals_scored', 'home_avg_goals_conceded', 'home_goal_diff',
    'home_wc_matches', 'home_wc_appearances', 'home_wc_titles',
    'away_fifa_rank', 'away_fifa_points', 'away_wc_win_rate',
    'away_avg_goals_scored', 'away_avg_goals_conceded', 'away_goal_diff',
    'away_wc_matches', 'away_wc_appearances', 'away_wc_titles',
    'rank_diff', 'points_diff', 'points_ratio',
    'win_rate_diff', 'goal_diff_diff', 'title_diff', 'experience_diff'
]

# Top Navigation
st.markdown("<h2 style='text-align: center; color: #00B2A9; margin-bottom: 1rem;'>⚽ FIFA World Cup 2026 Analytics Platform</h2>", unsafe_allow_html=True)

page = st.radio(
    "Navigate",
    ["🏠 Home", "🔮 Match Predictor", "🌍 Team Dashboard"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown("---")

# ── 1. HOME VIEW ──────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🏆 FIFA World Cup 2026 Analytics Platform")
    st.markdown("""
    Welcome to the **FIFA World Cup 2026 Analytics Platform**. This platform is a university AI semester project
    demonstrating the end-to-end Machine Learning lifecycle: Data Engineering, Classification Modeling, Model Explainability,
    Unsupervised Learning, and interactive visualizations.
    
    ### Platform Overview
    
    * **🔮 Module A — Match Predictor**:
      Predict the outcome of any scheduled WC 2026 group stage match. Uses a trained tree ensemble (**XGBoost**)
      to generate Home Win / Draw / Away Win probabilities, with **SHAP explanation waterfall charts** detailing the precise
      feature contributions driving the model's decisions.
    
    * **🌍 Module B — Team Intelligence Dashboard**:
      Explore individual team profiles through customizable **Radar Charts**, perform side-by-side **Head-to-Head Comparisons**,
      browse the full tournament **AI Strength Index leaderboard**, and explore team playing styles grouped using **K-Means clustering** and visualized via **PCA**.
    """)
    
    st.markdown("---")
    st.subheader("📊 Dataset & Model Statistics")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Historical Match Dataset", "436 rows", "1994–2022 WC")
    col2.metric("WC 2026 Scheduled Fixtures", "72 matches", "Group stage input")
    col3.metric("Aggregated Team Features", "66 countries", "Historical data")

# ── 2. MATCH PREDICTOR VIEW ───────────────────────────────────────
elif page == "🔮 Match Predictor":
    
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">FIFA World Cup 2026 Predictor</h1>
        <p class="hero-subtitle">AI-POWERED MATCH INTELLIGENCE PLATFORM</p>
    </div>
    """, unsafe_allow_html=True)
    
    pred_tab1, pred_tab2 = st.tabs(["📅 Group Stage Predictor", "🏆 Full Tournament Simulator"])
    
    with pred_tab1:
        st.markdown("Select an upcoming group stage match to run the prediction pipeline.")
        
        # Build match selector: show as "Date — Home Team vs Away Team"
        df_master['fixture_label'] = df_master['Date'] + ' — ' + df_master['home_team'] + ' vs ' + df_master['away_team']
        fixture = st.selectbox("Select Scheduled Match", sorted(df_master['fixture_label'].unique()))
        
        row = df_master[df_master['fixture_label'] == fixture].iloc[0]
        home = row['home_team']
        away = row['away_team']
        
        # Layout header
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.markdown(f"<div style='text-align: center;'><h3>🏠 {home}</h3><p>FIFA Rank: <b>#{int(row['home_fifa_rank'])}</b></p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<h2 style='text-align:center;padding-top:10px'>VS</h2>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div style='text-align: center;'><h3>✈️ {away}</h3><p>FIFA Rank: <b>#{int(row['away_fifa_rank'])}</b></p></div>", unsafe_allow_html=True)
            
        st.markdown("---")
        
        if st.button("🔮 Predict Match Outcome", type="primary", use_container_width=True):
            with st.spinner("Running classification model and SHAP attribution..."):
                result = predict_match(home, away, df_master, model, explainer, FEATURE_COLS)
                
            if result is None:
                st.error("Match fixture details could not be found.")
            else:
                # Dramatic Premium Prediction Card
                pred = result['prediction']
                
                winner_color = "#00B2A9" if pred == 'H' else "#FF4F81" if pred == 'A' else "#FFC857"
                confidence = max(result['home_win'], result['draw'], result['away_win'])
                
                # Predict Goals using team historical averages & model confidence
                h_hist = df_team[df_team['team'] == home]
                a_hist = df_team[df_team['team'] == away]
                
                h_avg_scored = h_hist['avg_goals_scored'].values[0] if not h_hist.empty else 1.0
                h_avg_conc = h_hist['avg_goals_conceded'].values[0] if not h_hist.empty else 1.0
                a_avg_scored = a_hist['avg_goals_scored'].values[0] if not a_hist.empty else 1.0
                a_avg_conc = a_hist['avg_goals_conceded'].values[0] if not a_hist.empty else 1.0
                
                h_goals = h_avg_scored * 0.6 + a_avg_conc * 0.4
                a_goals = a_avg_scored * 0.6 + h_avg_conc * 0.4
                
                home_win_prob = result['home_win'] / (result['home_win'] + result['away_win'] + 0.0001)
                sim_h_goals = int(round(h_goals * (1 + home_win_prob)))
                sim_a_goals = int(round(a_goals * (1 + (1 - home_win_prob))))
                
                if pred == 'H' and sim_h_goals <= sim_a_goals: sim_h_goals = sim_a_goals + 1
                if pred == 'A' and sim_a_goals <= sim_h_goals: sim_a_goals = sim_h_goals + 1
                if pred == 'D': sim_h_goals = sim_a_goals = max(sim_h_goals, sim_a_goals)
                
                html_card = f"""
<div class="prediction-hero">
<h2 style="color: {winner_color}; font-size: 3.5rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 2px;">
{'🏆 ' + home + ' WINS' if pred == 'H' else '🏆 ' + away + ' WINS' if pred == 'A' else '🤝 DRAW'}
</h2>
<h1 style="font-size: 6rem; margin: 1rem 0; text-shadow: 0 0 30px {winner_color}88;">{sim_h_goals} - {sim_a_goals}</h1>
<p style="font-size: 1.8rem; color: #cbd5e1; font-weight: 300;">AI Confidence: <b>{confidence:.1%}</b></p>
<div style="margin-top: 2rem; display: flex; align-items: center; justify-content: center; gap: 15px;">
<span style="color: #00B2A9; font-weight: bold; min-width: 150px; white-space: nowrap; text-align: right;">{home}</span>
<div style="flex-grow: 1; max-width: 400px; display: flex; height: 12px; border-radius: 6px; overflow: hidden; background: rgba(255,255,255,0.1);">
<div style="width: {result['home_win'] * 100}%; background: #00B2A9;" title="Home Win"></div>
<div style="width: {result['draw'] * 100}%; background: #FFC857;" title="Draw"></div>
<div style="width: {result['away_win'] * 100}%; background: #FF4F81;" title="Away Win"></div>
</div>
<span style="color: #FF4F81; font-weight: bold; min-width: 150px; white-space: nowrap; text-align: left;">{away}</span>
</div>
</div>
"""
                st.markdown(html_card, unsafe_allow_html=True)
                    
                # SHAP Waterfall explanation
                st.markdown("---")
                st.subheader("🔍 Explainable AI: SHAP Explanation Waterfall")
                st.caption("This chart details the features that pushed the prediction toward the selected outcome.")
                
                # Draw waterfall
                shap_exp = shap.Explanation(
                    values=result['shap_values'],
                    base_values=result['base_value'],
                    data=df_master[df_master['fixture_label'] == fixture][FEATURE_COLS].values[0],
                    feature_names=FEATURE_COLS
                )
                
                fig_shap, ax = plt.subplots(figsize=(10, 5))
                shap.plots.waterfall(shap_exp, max_display=10, show=False)
                plt.title(f"SHAP Decision Impact (Class: {pred})", fontsize=12, pad=15)
                st.pyplot(fig_shap)
                plt.close()
                
                # Team Comparison table
                st.markdown("---")
                st.subheader("📊 Team Attribute Comparison")
                
                # Fetch experience / appearance features safely
                home_app = row.get('home_wc_appearances_94_18', row.get('home_wc_appearances', 0))
                away_app = row.get('away_wc_appearances_94_18', row.get('away_wc_appearances', 0))
                
                comp_data = {
                    "Metric": ["FIFA Rank", "FIFA Points", "WC Win Rate", "Avg Goals Scored",
                               "Avg Goals Conceded", "WC Titles", "WC Appearances (Updated 2026)"],
                    home: [
                        int(row['home_fifa_rank']), f"{row['home_fifa_points']:.1f}",
                        f"{row['home_wc_win_rate']:.1%}", f"{row['home_avg_goals_scored']:.2f}",
                        f"{row['home_avg_goals_conceded']:.2f}", int(row['home_wc_titles']),
                        int(home_app)
                    ],
                    away: [
                        int(row['away_fifa_rank']), f"{row['away_fifa_points']:.1f}",
                        f"{row['away_wc_win_rate']:.1%}", f"{row['away_avg_goals_scored']:.2f}",
                        f"{row['away_avg_goals_conceded']:.2f}", int(row['away_wc_titles']),
                        int(away_app)
                    ]
                }
                st.dataframe(pd.DataFrame(comp_data).set_index("Metric"), use_container_width=True)
                
    with pred_tab2:
        st.markdown("Run a fully automated Monte Carlo-style bracket simulation based on our AI Model's strength metrics. The AI will pair the top 32 qualifying teams and autonomously predict every knockout round until a World Champion is crowned.")
        if st.button("🚀 Run Knockout Stage Simulation", type="primary", use_container_width=True):
            run_tournament_simulation(df_master, df_team, model, FEATURE_COLS)

# ── 3. TEAM INTELLIGENCE DASHBOARD VIEW ──────────────────────────
elif page == "🌍 Team Dashboard":
    st.title("🌍 Team Intelligence Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["📊 Team Explorer", "⚔️ Team Comparison", "🏆 Strength Leaderboard"])
    
    all_2026_teams = sorted(list(set(df_master['home_team']) | set(df_master['away_team'])))
    
    # ── TAB 1: TEAM EXPLORER ──────────────────────────────────────
    with tab1:
        selected_team = st.selectbox("Select a WC 2026 Qualified Team", all_2026_teams)
        
        col1, col2 = st.columns([3, 2])
        with col1:
            st.plotly_chart(radar_chart(selected_team, df_team), use_container_width=True, key="radar_explorer")
            
        with col2:
            row_hist = df_team[df_team['team'] == selected_team]
            if not row_hist.empty:
                r = row_hist.iloc[0]
                st.markdown(f"### 📊 Historical Profile ({selected_team})")
                
                # Check appearances column
                app_col = 'wc_appearances_94_22' if 'wc_appearances_94_22' in r else 'wc_appearances_94_18' if 'wc_appearances_94_18' in r else 'wc_appearances'
                
                m1, m2 = st.columns(2)
                m1.metric("WC Appearances (Updated 2026)", int(r[app_col]))
                m2.metric("WC Titles (All Time)", int(r['wc_titles']))
                
                m3, m4 = st.columns(2)
                m3.metric("Win Rate", f"{r['wc_win_rate']:.1%}")
                m4.metric("Avg Goals Scored", f"{r['avg_goals_scored']:.2f}")
                
                m5, m6 = st.columns(2)
                m5.metric("Avg Goals Conceded", f"{r['avg_goals_conceded']:.2f}")
                m6.metric("Goal Diff / Match", f"{r['goal_diff_per_match']:+.2f}")
            else:
                st.info(f"🔰 **{selected_team}** has no historical World Cup matches (1994–2018) in our dataset. They are making a modern debut or did not qualify in that window.")
        
        # Similar Teams & Style Clusters
        if selected_team in df_cluster['team'].values:
            st.markdown("---")
            st.subheader("🔗 Style Profile & Similarity")
            
            # Find similar teams
            CLUSTER_FEATURES = [
                'wc_win_rate', 'avg_goals_scored', 'avg_goals_conceded',
                'goal_diff_per_match', 'wc_appearances_94_22', 'wc_titles'
            ]
            team_row = df_cluster[df_cluster['team'] == selected_team][CLUSTER_FEATURES].values
            distances = df_cluster[CLUSTER_FEATURES].apply(
                lambda r_val: np.sqrt(((r_val.values - team_row[0]) ** 2).sum()), axis=1
            )
            similar = df_cluster.assign(distance=distances).sort_values('distance')[1:6]
            
            st.markdown(f"#### Most Similar Teams to **{selected_team}**")
            st.dataframe(
                similar[['team', 'cluster_name', 'wc_win_rate', 'wc_titles', 'distance']].rename(
                    columns={
                        'team': 'Team',
                        'cluster_name': 'Tactical Style Group',
                        'wc_win_rate': 'Win Rate',
                        'wc_titles': 'WC Titles',
                        'distance': 'Euclidean Distance'
                    }
                ),
                use_container_width=True
            )
            
            # Scatter Plot (PCA)
            st.markdown("#### Style Scatter Plot (PCA Representation)")
            st.caption("Visualizing K-Means clusters of playing style projected onto the first two Principal Components.")
            
            # Mark selected team
            df_cluster['is_selected'] = df_cluster['team'] == selected_team
            df_cluster['size_marker'] = df_cluster['is_selected'].map({True: 15, False: 8})
            
            fig_scatter = px.scatter(
                df_cluster, x='pca1', y='pca2',
                color='cluster_name', hover_name='team',
                size='size_marker', size_max=15,
                labels={'pca1': 'Principal Component 1', 'pca2': 'Principal Component 2', 'cluster_name': 'Style Group'},
                title=f"Tactical Clusters (Highlighted: {selected_team})"
            )
            fig_scatter.update_layout(
                height=450,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_scatter, use_container_width=True, key="style_scatter_plot")
            
            # Clean up column
            df_cluster.drop(columns=['is_selected', 'size_marker'], inplace=True, errors='ignore')
            
    # ── TAB 2: TEAM COMPARISON ────────────────────────────────────
    with tab2:
        c1, c2 = st.columns(2)
        team_a = c1.selectbox("Team A", all_2026_teams, key='cmp_a')
        team_b = c2.selectbox("Team B", all_2026_teams, index=min(1, len(all_2026_teams)-1), key='cmp_b')
        
        col_rad_a, col_rad_b = st.columns(2)
        col_rad_a.plotly_chart(radar_chart(team_a, df_team), use_container_width=True, key="radar_comp_a")
        col_rad_b.plotly_chart(radar_chart(team_b, df_team), use_container_width=True, key="radar_comp_b")
        
        # Side-by-side comparison chart
        st.subheader("📊 Side-by-Side Performance Comparison")
        metrics = ['wc_win_rate', 'avg_goals_scored', 'avg_goals_conceded', 'goal_diff_per_match']
        labels = ['Win Rate', 'Avg Goals Scored', 'Avg Goals Conceded', 'Goal Diff/Match']
        
        row_a = df_team[df_team['team'] == team_a].iloc[0] if team_a in df_team['team'].values else None
        row_b = df_team[df_team['team'] == team_b].iloc[0] if team_b in df_team['team'].values else None
        
        vals_a = [float(row_a[m]) if row_a is not None else 0.0 for m in metrics]
        vals_b = [float(row_b[m]) if row_b is not None else 0.0 for m in metrics]
        
        fig_bar = go.Figure(data=[
            go.Bar(name=team_a, x=labels, y=vals_a, marker_color='#3b82f6'),
            go.Bar(name=team_b, x=labels, y=vals_b, marker_color='#f97316')
        ])
        fig_bar.update_layout(
            barmode='group',
            title='Head-to-Head Key Match Metrics',
            height=350,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="comp_bar_chart")

    # ── TAB 3: STRENGTH INDEX LEADERBOARD ──────────────────────────
    with tab3:
        st.subheader("🏆 Tournament Strength Index Leaderboard")
        st.markdown("All 48 World Cup 2026 qualified teams ranked by an AI-weighted strength index incorporating FIFA rankings, historic tournament results, and recent form.")
        
        strengths = []
        for t in all_2026_teams:
            # Get FIFA rank safely
            team_fixture = df_master[df_master['home_team'] == t]
            if team_fixture.empty:
                team_fixture = df_master[df_master['away_team'] == t]
            
            if not team_fixture.empty:
                r_val = team_fixture.iloc[0]
                rank = int(r_val['home_fifa_rank']) if r_val['home_team'] == t else int(r_val['away_fifa_rank'])
            else:
                rank = 999
                
            strengths.append({
                'Team': t,
                'Strength Index': compute_strength_index(t, df_master),
                'FIFA Rank': rank
            })
            
        df_strengths = pd.DataFrame(strengths).sort_values('Strength Index', ascending=False).reset_index(drop=True)
        df_strengths.index += 1  # 1-indexed leaderboard
        
        # Display table
        st.dataframe(df_strengths, use_container_width=True)
        
        # Display plot of top 20
        st.markdown("---")
        st.subheader("📊 Top 20 Teams by Strength Score")
        fig_ldr = px.bar(
            df_strengths.head(20), x='Team', y='Strength Index',
            color='Strength Index', color_continuous_scale='Blues',
            labels={'Team': 'Country', 'Strength Index': 'Composite Score'},
            title='Top 20 Teams Leaderboard'
        )
        fig_ldr.update_xaxes(tickangle=45)
        fig_ldr.update_layout(
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_ldr, use_container_width=True, key="leaderboard_bar_chart")
