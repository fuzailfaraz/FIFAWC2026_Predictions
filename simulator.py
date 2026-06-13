import streamlit as st
import pandas as pd
import numpy as np

def predict_knockout_match(home, away, df_team, df_master, model, feature_cols, is_group_stage=False):
    h_hist = df_team[df_team['team'] == home]
    a_hist = df_team[df_team['team'] == away]
    
    if h_hist.empty or a_hist.empty:
        # Fallback for missing data
        w = home if np.random.rand() > 0.5 else away
        return {"home": home, "away": away, "winner": w, "home_score": 1, "away_score": 0, "prob": 0.5, "reason": "Insufficient data"}
        
    h_hist = h_hist.iloc[0]
    a_hist = a_hist.iloc[0]
    
    h_master = df_master[(df_master['home_team'] == home) | (df_master['away_team'] == home)]
    a_master = df_master[(df_master['home_team'] == away) | (df_master['away_team'] == away)]
    
    h_rank = 50; h_pts = 1500
    if not h_master.empty:
        r = h_master.iloc[0]
        h_rank = r['home_fifa_rank'] if r['home_team'] == home else r['away_fifa_rank']
        h_pts = r['home_fifa_points'] if r['home_team'] == home else r['away_fifa_points']
        
    a_rank = 50; a_pts = 1500
    if not a_master.empty:
        r = a_master.iloc[0]
        a_rank = r['home_fifa_rank'] if r['home_team'] == away else r['away_fifa_rank']
        a_pts = r['home_fifa_points'] if r['home_team'] == away else r['away_fifa_points']

    row = {
        'home_fifa_rank': h_rank, 'home_fifa_points': h_pts,
        'home_wc_win_rate': h_hist['wc_win_rate'],
        'home_avg_goals_scored': h_hist['avg_goals_scored'],
        'home_avg_goals_conceded': h_hist['avg_goals_conceded'],
        'home_goal_diff': h_hist['goal_diff_per_match'],
        'home_wc_matches': h_hist['wc_matches_played'],
        'home_wc_appearances': h_hist.get('wc_appearances_94_22', h_hist.get('wc_appearances_94_18', 0)),
        'home_wc_titles': h_hist['wc_titles'],
        
        'away_fifa_rank': a_rank, 'away_fifa_points': a_pts,
        'away_wc_win_rate': a_hist['wc_win_rate'],
        'away_avg_goals_scored': a_hist['avg_goals_scored'],
        'away_avg_goals_conceded': a_hist['avg_goals_conceded'],
        'away_goal_diff': a_hist['goal_diff_per_match'],
        'away_wc_matches': a_hist['wc_matches_played'],
        'away_wc_appearances': a_hist.get('wc_appearances_94_22', a_hist.get('wc_appearances_94_18', 0)),
        'away_wc_titles': a_hist['wc_titles'],
    }
    
    row['rank_diff'] = row['home_fifa_rank'] - row['away_fifa_rank']
    row['points_diff'] = row['home_fifa_points'] - row['away_fifa_points']
    row['points_ratio'] = row['home_fifa_points'] / max(row['away_fifa_points'], 1)
    row['win_rate_diff'] = row['home_wc_win_rate'] - row['away_wc_win_rate']
    row['goal_diff_diff'] = row['home_goal_diff'] - row['away_goal_diff']
    row['title_diff'] = row['home_wc_titles'] - row['away_wc_titles']
    row['experience_diff'] = row['home_wc_appearances'] - row['away_wc_appearances']
    
    X = pd.DataFrame([row])[feature_cols]
    proba = model.predict_proba(X)[0]
    
    prob_h_model = proba[2]
    prob_a_model = proba[0]
    
    # --- ELO RATINGS BLENDING ---
    try:
        elo_df = pd.read_csv('data/elo_ratings_df.csv')
        h_elo = elo_df[elo_df['team'] == home]['elo_rating'].values[0] if home in elo_df['team'].values else 1500
        a_elo = elo_df[elo_df['team'] == away]['elo_rating'].values[0] if away in elo_df['team'].values else 1500
    except:
        h_elo = 1500; a_elo = 1500
        
    elo_diff = h_elo - a_elo
    prob_h_elo = 1 / (1 + 10 ** (-elo_diff / 400))
    prob_a_elo = 1 - prob_h_elo
    
    # 70% weight to Elo for realistic recent form
    prob_h = prob_h_model * 0.3 + prob_h_elo * 0.7
    prob_a = prob_a_model * 0.3 + prob_a_elo * 0.7
    
    home_win_prob = prob_h / (prob_h + prob_a + 0.0001)
    
    # Predict Scores
    h_goals = h_hist['avg_goals_scored'] * 0.6 + a_hist['avg_goals_conceded'] * 0.4
    a_goals = a_hist['avg_goals_scored'] * 0.6 + h_hist['avg_goals_conceded'] * 0.4
    
    if home_win_prob > 0.5:
        h_goals += (home_win_prob - 0.5) * 3
        a_goals -= (home_win_prob - 0.5) * 2
    else:
        a_goals += ((1 - home_win_prob) - 0.5) * 3
        h_goals -= ((1 - home_win_prob) - 0.5) * 2
        
    h_score = int(round(max(0, h_goals)))
    a_score = int(round(max(0, a_goals)))
    
    if h_score == a_score and not is_group_stage:
        if home_win_prob > 0.5:
            h_score += 1
        else:
            a_score += 1

    if h_score > a_score:
        winner = home
    elif a_score > h_score:
        winner = away
    else:
        winner = "DRAW"
    
    # Reason
    reasons = []
    
    w_elo = h_elo if winner == home else a_elo
    l_elo = a_elo if winner == home else h_elo
    if w_elo > l_elo + 20:
        reasons.append(f"Higher Recent Elo ({int(w_elo)} vs {int(l_elo)})")
    
    w_rank = h_rank if winner == home else a_rank
    l_rank = a_rank if winner == home else h_rank
    if w_rank < l_rank and len(reasons) < 2:
        reasons.append(f"Higher FIFA Rank (#{int(w_rank)} vs #{int(l_rank)})")
    
    w_rate = h_hist['wc_win_rate'] if winner == home else a_hist['wc_win_rate']
    l_rate = a_hist['wc_win_rate'] if winner == home else h_hist['wc_win_rate']
    if w_rate > l_rate + 0.05 and len(reasons) < 2:
        reasons.append(f"Superior WC Win Rate ({w_rate:.0%} > {l_rate:.0%})")
        
    w_att = h_hist['avg_goals_scored'] if winner == home else a_hist['avg_goals_scored']
    l_att = a_hist['avg_goals_scored'] if winner == home else h_hist['avg_goals_scored']
    if w_att > l_att + 0.2 and len(reasons) < 2:
        reasons.append(f"More potent attack ({w_att:.1f} vs {l_att:.1f} avg goals)")
        
    if not reasons:
        reasons.append(f"AI Probability Edge ({max(prob_h, prob_a):.1%})")

    return {
        "home": home, "away": away,
        "home_score": h_score, "away_score": a_score,
        "winner": winner,
        "prob": max(prob_h, prob_a),
        "reason": ", ".join(reasons[:2])
    }

def run_tournament_simulation(df_master, df_team, model, feature_cols):
    st.markdown("<h2 style='text-align: center; font-size: 3rem; color: #00ff00;'>🏆 WORLD CUP 2026 FULL SIMULATION</h2>", unsafe_allow_html=True)
    
    # Load Group and Knockout formats
    try:
        df_groups = pd.read_excel('data/Groups.xlsx')
        df_knockouts = pd.read_excel('data/Knockouts.xlsx')
    except Exception as e:
        st.error(f"Could not load Groups.xlsx or Knockouts.xlsx: {e}")
        return
        
    df_groups['Team'] = df_groups['Team'].replace({
        'South Korea': 'Korea Republic',
        'Ivory Coast': "Côte d'Ivoire",
        'DR Congo': 'Congo DR',
        'Cape Verde': 'Cabo Verde'
    })
    
    # Simulate Group Stage
    st.markdown("<h3 style='text-align: center; color: #FFC857; margin-top: 2rem;'>⚽ GROUP STAGE MATCHES & STANDINGS</h3>", unsafe_allow_html=True)
    
    # Setup standings
    standings = {}
    for _, row in df_groups.iterrows():
        t = row['Team']
        grp = row['Group']
        standings[t] = {'Team': t, 'Group': grp, 'Pts': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'GD': 0}
        
    # Simulate all group stage matches (from master dataset)
    for _, row in df_master[df_master['Round'] == 'Group stage'].iterrows():
        h = row['home_team']
        a = row['away_team']
        if h not in standings or a not in standings:
            continue
            
        res = predict_knockout_match(h, a, df_team, df_master, model, feature_cols, is_group_stage=True)
        h_score = res['home_score']
        a_score = res['away_score']
        
        standings[h]['GF'] += h_score
        standings[h]['GA'] += a_score
        standings[h]['GD'] += (h_score - a_score)
        standings[a]['GF'] += a_score
        standings[a]['GA'] += h_score
        standings[a]['GD'] += (a_score - h_score)
        
        if h_score > a_score:
            standings[h]['W'] += 1
            standings[h]['Pts'] += 3
            standings[a]['L'] += 1
        elif a_score > h_score:
            standings[a]['W'] += 1
            standings[a]['Pts'] += 3
            standings[h]['L'] += 1
        else:
            standings[h]['D'] += 1
            standings[h]['Pts'] += 1
            standings[a]['D'] += 1
            standings[a]['Pts'] += 1
            
    # Display Group Tables
    df_st = pd.DataFrame(standings.values())
    
    group_labels = sorted(df_st['Group'].unique())
    cols = st.columns(4)
    for i, grp_lbl in enumerate(group_labels):
        grp_df = df_st[df_st['Group'] == grp_lbl].sort_values(by=['Pts', 'GD', 'GF'], ascending=False).reset_index(drop=True)
        grp_df.index += 1
        with cols[i % 4]:
            st.markdown(f"**Group {grp_lbl}**")
            st.dataframe(grp_df[['Team', 'Pts', 'GD']], use_container_width=True)
            
    # Determine advancing teams
    pos_mapping = {}
    third_placed = []
    
    for grp in group_labels:
        grp_df = df_st[df_st['Group'] == grp].sort_values(by=['Pts', 'GD', 'GF'], ascending=False).reset_index(drop=True)
        pos_mapping[f"1{grp}"] = grp_df.iloc[0]['Team']
        pos_mapping[f"2{grp}"] = grp_df.iloc[1]['Team']
        third_placed.append(grp_df.iloc[2])
        
    third_df = pd.DataFrame(third_placed).sort_values(by=['Pts', 'GD', 'GF'], ascending=False)
    best_8_third = third_df.iloc[0:8]['Team'].tolist()
    
    # Fill remaining missing runners-up logic dynamically
    b3_pool = best_8_third.copy()
    b3_pool.extend([pos_mapping.get('2I', ''), pos_mapping.get('2K', ''), pos_mapping.get('2L', '')])
    b3_pool = [x for x in b3_pool if x] # remove empty if they were missing
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #FFC857; margin-top: 2rem;'>⚔️ KNOCKOUT STAGE BRACKET</h3>", unsafe_allow_html=True)

    # Define Left and Right Matchups explicitly based on the official 2026 split bracket format
    r32_left_matchups = [
        ("1E", "B3_1"), ("1I", "B3_2"), ("2A", "2B"), ("1F", "2C"),
        ("2K", "2L"), ("1H", "2J"), ("1D", "B3_3"), ("1G", "B3_4")
    ]
    r32_right_matchups = [
        ("1C", "2F"), ("2E", "2I"), ("1A", "B3_5"), ("1L", "B3_6"),
        ("1J", "2H"), ("2D", "2G"), ("1B", "B3_7"), ("1K", "B3_8")
    ]
    
    b3_index = 0
    def get_team(pos):
        nonlocal b3_index
        if pos.startswith("B3_"):
            if b3_index < len(b3_pool):
                t = b3_pool[b3_index]
                b3_index += 1
                return t
            return "TBD"
        return pos_mapping.get(pos, pos)
        
    # Simulate Bracket Stages
    r32_l = [predict_knockout_match(get_team(m1), get_team(m2), df_team, df_master, model, feature_cols) for m1, m2 in r32_left_matchups]
    r32_r = [predict_knockout_match(get_team(m1), get_team(m2), df_team, df_master, model, feature_cols) for m1, m2 in r32_right_matchups]
    
    r16_l = [predict_knockout_match(r32_l[i]['winner'], r32_l[i+1]['winner'], df_team, df_master, model, feature_cols) for i in range(0, 8, 2)]
    r16_r = [predict_knockout_match(r32_r[i]['winner'], r32_r[i+1]['winner'], df_team, df_master, model, feature_cols) for i in range(0, 8, 2)]
    
    qf_l = [predict_knockout_match(r16_l[i]['winner'], r16_l[i+1]['winner'], df_team, df_master, model, feature_cols) for i in range(0, 4, 2)]
    qf_r = [predict_knockout_match(r16_r[i]['winner'], r16_r[i+1]['winner'], df_team, df_master, model, feature_cols) for i in range(0, 4, 2)]
    
    sf_l = [predict_knockout_match(qf_l[0]['winner'], qf_l[1]['winner'], df_team, df_master, model, feature_cols)]
    sf_r = [predict_knockout_match(qf_r[0]['winner'], qf_r[1]['winner'], df_team, df_master, model, feature_cols)]
    
    final = [predict_knockout_match(sf_l[0]['winner'], sf_r[0]['winner'], df_team, df_master, model, feature_cols)]
    
    # HTML Visual Bracket Generator (Split Left-Right Converging Layout)
    bracket_html = """
    <style>
    .match-box { position: relative; }
    .left-side .match-box::after {
        content: ''; position: absolute; right: -25px; top: 50%; width: 25px; height: 2px; background: rgba(255, 200, 87, 0.5); z-index: -1;
    }
    .right-side .match-box::before {
        content: ''; position: absolute; left: -25px; top: 50%; width: 25px; height: 2px; background: rgba(255, 200, 87, 0.5); z-index: -1;
    }
    .center-side .match-box::after, .center-side .match-box::before { display: none; }
    </style>
    <div style="display: flex; flex-direction: row; width: 100%; min-width: 2600px; height: auto; min-height: 1800px; background: rgba(10, 25, 47, 0.5); backdrop-filter: blur(15px) saturate(180%); font-family: 'Outfit', sans-serif; overflow-x: auto; padding: 20px; border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.1); gap: 20px;">
    """
    
    def render_col(matches, title, side="center", is_final=False):
        col_html = f"<div class='{side}' style='display: flex; flex-direction: column; justify-content: space-around; flex: 1; padding: 5px; position: relative;'>"
        col_html += f"<h3 style='text-align: center; color: #FFC857; font-family: \"Bebas Neue\", sans-serif; font-size: 2rem; letter-spacing: 2px; margin-bottom: 20px; text-transform: uppercase;'>{title}</h3>"
        
        for match in matches:
            h_winner = match['home'] == match['winner']
            a_winner = match['away'] == match['winner']
            h_color, a_color = ("#00B2A9", "#94a3b8") if h_winner else ("#94a3b8", "#00B2A9")
            h_weight, a_weight = ("900", "400") if h_winner else ("400", "900")
            h_ind, a_ind = ("#00B2A9", "#475569") if h_winner else ("#475569", "#FF4F81")
            
            box_style = "background: rgba(10, 25, 47, 0.65); backdrop-filter: blur(12px) saturate(180%); border-radius: 12px; padding: 12px; margin: 10px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.3); transition: transform 0.2s;"
            if is_final:
                box_style += "border: 2px solid rgba(212, 175, 55, 0.8); box-shadow: 0 0 25px rgba(212, 175, 55, 0.4);"
            else:
                box_style += "border: 1px solid rgba(255, 255, 255, 0.1);"
                
            box_html = f"""
            <div class="match-box" style="{box_style}" onmouseover="this.style.transform='scale(1.03)';" onmouseout="this.style.transform='scale(1)';">
                <div style="display: flex; justify-content: space-between; font-size: 1.4rem; font-weight: {h_weight}; color: {h_color};">
                    <div><span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:{h_ind}; margin-right:8px;"></span>{match['home']}</div>
                    <span>{match['home_score']}</span>
                </div>
                <div style="height: 1px; background: rgba(255,255,255,0.1); margin: 8px 0;"></div>
                <div style="display: flex; justify-content: space-between; font-size: 1.4rem; font-weight: {a_weight}; color: {a_color};">
                    <div><span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:{a_ind}; margin-right:8px;"></span>{match['away']}</div>
                    <span>{match['away_score']}</span>
                </div>
                <div style="font-size: 1rem; color: #94a3b8; margin-top: 10px; border-top: 1px dashed rgba(255,255,255,0.1); padding-top: 6px; text-align: center;">
                    ⚡ <b>{match['prob']:.1%}</b> | {match['reason']}
                </div>
            </div>
            """
            col_html += box_html
        col_html += "</div>"
        return col_html
        
    bracket_html += render_col(r32_l, "Round 32", "left-side")
    bracket_html += render_col(r16_l, "Round 16", "left-side")
    bracket_html += render_col(qf_l, "Quarter-final", "left-side")
    bracket_html += render_col(sf_l, "Semi-final", "left-side")
    bracket_html += render_col(final, "WORLD CHAMPIONSHIP", "center-side", is_final=True)
    bracket_html += render_col(sf_r, "Semi-final", "right-side")
    bracket_html += render_col(qf_r, "Quarter-final", "right-side")
    bracket_html += render_col(r16_r, "Round 16", "right-side")
    bracket_html += render_col(r32_r, "Round 32", "right-side")
    
    bracket_html += "</div>"
    
    champ = final[0]['winner'] if final else "TBD"
    bracket_html += f"""
    <div style='background: rgba(10, 25, 47, 0.7); border: 2px solid #FFC857; padding: 3rem; border-radius: 2rem; text-align: center; margin-top: 2rem; box-shadow: 0 10px 40px rgba(255, 200, 87, 0.2);'>
        <h2 style='color: #FF4F81; font-family: "Bebas Neue", sans-serif; font-size: 3rem; margin: 0; letter-spacing: 2px;'>WORLD CUP WINNER PREDICTION</h2>
        <h1 style='color: #FFC857; font-family: "Bebas Neue", sans-serif; font-size: 6rem; margin: 1rem 0;'>🏆 {champ.upper()} 🏆</h1>
    </div>
    """
    
    st.components.v1.html(bracket_html, height=2100, scrolling=True)
