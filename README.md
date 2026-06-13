# 🏆 FIFA World Cup 2026 Analytics Platform

An end-to-end AI/ML analytics platform that predicts match outcomes, simulates the full 48-team tournament bracket, and provides deep team-level analytics for the **FIFA World Cup 2026 (USA / Canada / Mexico)**.

Built as a university semester project for **BS Data Science — Artificial Intelligence & Machine Learning**, this platform demonstrates the complete ML lifecycle: data ingestion, feature engineering, model training, explainable AI, and interactive dashboard deployment via **Streamlit**.

---

## ✨ Features

- **🔮 AI Match Prediction** — Predicts Home Win / Draw / Away Win and an estimated scoreline for any of the 72 scheduled group stage fixtures, with a confidence score.
- **🏟️ Full Tournament Simulation** — Simulates the entire 48-team bracket end-to-end (Group Stage → Round of 32 → Round of 16 → Quarter-Finals → Semi-Finals → Final) and crowns an AI-predicted World Champion.
- **📊 Team Intelligence Dashboard** — Radar charts, head-to-head comparisons, a global strength leaderboard, and K-Means playing-style clusters with PCA visualization.
- **🧠 Explainable AI** — Every prediction comes with a SHAP waterfall chart showing exactly which features pushed the result toward each team.
- **⚡ Live Strength Signal** — Blends historical World Cup data with live Elo ratings (scraped from eloratings.net) so predictions reflect current squad form, not just past tournaments.

---

## 🧰 Tech Stack

| Technology | Role |
|---|---|
| Python 3.13 | Core language |
| Streamlit | Interactive web UI / dashboard |
| Pandas / NumPy | Data manipulation & numerical computing |
| XGBoost | Primary classification model (match outcome prediction) |
| scikit-learn | K-Means clustering, PCA, preprocessing |
| SHAP | Explainable AI (feature contribution waterfall charts) |
| Plotly / Matplotlib | Interactive & static visualizations |
| Requests | Live Elo ratings scraper (eloratings.net) |
| openpyxl | Reading official group & bracket structure files |

---

## 🤖 How the Predictions Work

Each prediction blends two complementary signals:

1. **XGBoost Classifier** — trained on 30+ years of World Cup match data (1994–2022), capturing historical pedigree, title counts, and tournament-specific patterns.
2. **Elo Rating System** — a live, continuously-updated measure of each team's current real-world strength based on every international match played.

```
Final_Prob_Home = (XGBoost_Prob_Home × 0.30) + (Elo_Prob_Home × 0.70)
Final_Prob_Away = (XGBoost_Prob_Away × 0.30) + (Elo_Prob_Away × 0.70)
```

The 70/30 Elo-weighted blend ensures recent form dominates while historical pedigree still provides a meaningful correction. Every prediction is paired with a **SHAP waterfall chart** that explains *why* the model leaned the way it did.

---

## 📁 Project Structure

```
WC/
├── app.py                          # Main Streamlit UI application
├── predictor.py                    # Single-match prediction engine (XGBoost + Elo blend)
├── simulator.py                    # Full knockout bracket simulation engine
├── player_analysis.py              # Radar charts & Strength Index computation
├── elo_pipeline.py                 # Live Elo ratings data scraper (eloratings.net)
├── run_wc_simulation.py            # CLI script to run a full simulation in the terminal
├── models/
│   ├── xgb_model.pkl               # Trained XGBoost classifier
│   └── shap_explainer.pkl          # SHAP TreeExplainer object
├── data/
│   ├── wc2026_master_dataset_updated.csv      # All 72 group stage fixtures with features
│   ├── wc_training_dataset_updated.csv        # Historical WC match data (1994–2022)
│   ├── team_historical_features_updated.csv   # Per-team aggregated WC stats
│   ├── elo_ratings_df.csv                     # Live Elo ratings snapshot (245 teams)
│   ├── Groups.xlsx                            # Official 2026 group assignments
│   └── Knockouts.xlsx                         # Official 2026 bracket structure
└── notebooks/
    ├── 01_data_exploration.ipynb   # EDA and data understanding
    ├── 02_model_training.ipynb     # Full XGBoost training pipeline
    └── 03_clustering.ipynb         # K-Means team clustering analysis
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.13 (or later)
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/wc2026-analytics-platform.git
cd wc2026-analytics-platform

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app.py
```

### Run a Simulation from the CLI

```bash
python run_wc_simulation.py
```

### Refresh Live Elo Ratings (optional)

```bash
python elo_pipeline.py
```

---

## 📊 Datasets

| File | Description |
|---|---|
| `wc_training_dataset_updated.csv` | ~436 historical WC matches (1994–2022) with home/away features and result labels |
| `wc2026_master_dataset_updated.csv` | Pre-computed feature vectors for all 72 WC 2026 group stage fixtures |
| `team_historical_features_updated.csv` | Aggregated per-team WC statistics since 1994 |
| `elo_ratings_df.csv` | Live Elo ratings snapshot for 245 national teams |
| `Groups.xlsx` / `Knockouts.xlsx` | Official 2026 group assignments and bracket structure |

---

## ⚠️ Limitations

- Operates at the **team level** — does not account for individual player form, injuries, or suspensions.
- No nuanced **home-advantage** modeling across the three host nations.
- Pre-computed features only exist for the 72 group stage fixtures; knockout matches are evaluated using aggregated historical team data.
- "Best 3rd place" qualification uses a simplified points/GD/GF tiebreaker rather than FIFA's full disciplinary-record rules.
- Elo ratings are a static snapshot — re-run `elo_pipeline.py` before major tournaments for up-to-date results.

---

## 🔭 Future Enhancements

- Player-level data integration (injuries, squad depth, recent form) via APIs like WhoScored or FBRef
- Monte Carlo simulation of the bracket (10,000+ runs) for probabilistic championship odds
- Live, in-match prediction updates via real-time event streams
- LSTM-based time-series modeling of team momentum
- Ensemble stacking of XGBoost, Random Forest, and Logistic Regression via a meta-learner

---

## 📄 License

This project is intended for academic and educational purposes. Feel free to fork, adapt, and build upon it.

---

## 🙏 Acknowledgements

- Elo ratings data courtesy of [eloratings.net](https://www.eloratings.net)
- Built with [Streamlit](https://streamlit.io), [XGBoost](https://xgboost.readthedocs.io), and [SHAP](https://shap.readthedocs.io)
