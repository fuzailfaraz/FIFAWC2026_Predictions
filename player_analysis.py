import pandas as pd
import numpy as np
import plotly.graph_objects as go

def compute_strength_index(team_name: str, df_master: pd.DataFrame) -> float:
    """
    Composite 0-100 score based on available features in wc2026_master_dataset.
    Used in the Tournament Strength Leaderboard.
    """
    row = df_master[df_master['home_team'] == team_name]
    if row.empty:
        row = df_master[df_master['away_team'] == team_name]
    if row.empty:
        return 0.0

    row = row.iloc[0]
    side = 'home' if row['home_team'] == team_name else 'away'

    # Retrieve columns dynamically with fallbacks to avoid key errors
    fifa_rank = float(row.get(f'{side}_fifa_rank', 100))
    fifa_points = float(row.get(f'{side}_fifa_points', 0))
    wc_win_rate = float(row.get(f'{side}_wc_win_rate', 0))
    wc_titles = float(row.get(f'{side}_wc_titles', 0))
    
    # Try appearances_94_18, then standard wc_appearances
    wc_app = float(row.get(f'{side}_wc_appearances_94_18', row.get(f'{side}_wc_appearances', 0)))

    # Weighted composite — weights reflect feature importance from SHAP
    rank_score     = max(0.0, 100.0 - fifa_rank)      # higher rank = lower number
    points_score   = fifa_points / 19.0           # max ~1900 pts → scale to 100
    win_rate_score = wc_win_rate * 100.0
    title_score    = wc_titles * 10.0               # each title = 10 pts
    exp_score      = wc_app * 5.0     # each tournament = 5 pts

    composite = (0.35 * points_score +
                 0.25 * rank_score +
                 0.20 * win_rate_score +
                 0.12 * title_score +
                 0.08 * exp_score)
    return round(float(composite), 2)


def radar_chart(team_name: str, df_team: pd.DataFrame) -> go.Figure:
    """Radar chart of a team's WC stats. Returns Plotly figure."""
    RADAR_FEATURES   = ['wc_win_rate', 'avg_goals_scored', 'avg_goals_conceded',
                        'wc_appearances_94_18', 'wc_titles']
    RADAR_LABELS     = ['Win Rate', 'Goals Scored', 'Goals Conceded',
                        'WC Appearances', 'WC Titles']
    RADAR_MAX        = [1.0, 3.0, 3.0, 7.0, 5.0]   # normalisation maxes

    row = df_team[df_team['team'] == team_name]

    if row.empty:
        # Team has no WC history — show empty radar with message
        fig = go.Figure()
        fig.add_annotation(text=f"{team_name} has no WC history (1994–2018)",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=14))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=False),
                angularaxis=dict(visible=False)
            ),
            title=f"{team_name} — No WC History",
            height=400
        )
        return fig

    # Map features with fallback in case column names differ slightly
    values = []
    for f, mx in zip(RADAR_FEATURES, RADAR_MAX):
        col_val = row.get(f, row.get(f.replace('_94_18', ''), pd.Series([0.0])))
        val = float(col_val.values[0])
        values.append(min(val / mx, 1.0))
        
    values += [values[0]]                  # close the polygon
    labels  = RADAR_LABELS + [RADAR_LABELS[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values, theta=labels,
        fill='toself', name=team_name,
        fillcolor='rgba(0, 178, 169, 0.4)',
        line=dict(color='rgb(0, 178, 169)', width=3)
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.2)', linecolor='rgba(255,255,255,0.2)'),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.2)', linecolor='rgba(255,255,255,0.2)')
        ),
        title=f"{team_name} — WC Performance Radar",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig
