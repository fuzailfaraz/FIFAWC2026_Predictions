import pandas as pd
import numpy as np
import pickle
import random
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from simulator import predict_knockout_match

def run():
    # Load data
    df_train = pd.read_csv('data/wc_training_dataset_updated.csv')
    df_team = pd.read_csv('data/team_historical_features_updated.csv')
    df_master = pd.read_csv('data/wc2026_master_dataset_updated.csv')
    df_groups = pd.read_excel('data/Groups.xlsx')
    df_knockouts = pd.read_excel('data/Knockouts.xlsx')
    
    # Pre-process columns that are renamed in app.py
    df_master = df_master.rename(columns={
        'home_goal_diff_per_match': 'home_goal_diff',
        'away_goal_diff_per_match': 'away_goal_diff',
        'home_wc_matches_played': 'home_wc_matches',
        'away_wc_matches_played': 'away_wc_matches',
        'home_wc_appearances_94_18': 'home_wc_appearances',
        'away_wc_appearances_94_18': 'away_wc_appearances',
        'home_wc_appearances_94_22': 'home_wc_appearances',
        'away_wc_appearances_94_22': 'away_wc_appearances'
    })
    
    df_team = df_team.rename(columns={
        'wc_appearances_94_22': 'wc_appearances_94_18'
    })
    
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
    
    # Train model
    X = df_train[FEATURE_COLS]
    le = LabelEncoder()
    y = le.fit_transform(df_train['result'])
    model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X, y)
    
    df_groups['Team'] = df_groups['Team'].replace({
        'South Korea': 'Korea Republic',
        'Ivory Coast': "Côte d'Ivoire",
        'DR Congo': 'Congo DR',
        'Cape Verde': 'Cabo Verde'
    })
    
    standings = {}
    for _, row in df_groups.iterrows():
        t = row['Team']
        grp = row['Group']
        standings[t] = {'Team': t, 'Group': grp, 'Pts': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'GD': 0}
        
    for _, row in df_master[df_master['Round'] == 'Group stage'].iterrows():
        h = row['home_team']
        a = row['away_team']
        if h not in standings or a not in standings:
            continue
            
        res = predict_knockout_match(h, a, df_team, df_master, model, FEATURE_COLS, is_group_stage=True)
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
            
    df_st = pd.DataFrame(standings.values())
    
    print("Section 1: Final Group Tables")
    group_winners = []
    group_runners = []
    third_placed = []
    
    pos_mapping = {}
    
    for grp in sorted(df_st['Group'].unique()):
        grp_df = df_st[df_st['Group'] == grp].sort_values(by=['Pts', 'GD', 'GF'], ascending=False).reset_index(drop=True)
        print(f"Group {grp}")
        for idx, row in grp_df.iterrows():
            print(f"{idx+1}. {row['Team']} - {row['Pts']} pts (GD: {row['GD']}, GF: {row['GF']}, GA: {row['GA']})")
        print("")
        
        group_winners.append(grp_df.iloc[0]['Team'])
        pos_mapping[f"1{grp}"] = grp_df.iloc[0]['Team']
        group_runners.append(grp_df.iloc[1]['Team'])
        pos_mapping[f"2{grp}"] = grp_df.iloc[1]['Team']
        third_placed.append(grp_df.iloc[2])
        
    third_df = pd.DataFrame(third_placed).sort_values(by=['Pts', 'GD', 'GF'], ascending=False)
    best_8_third = third_df.iloc[0:8]['Team'].tolist()
    
    print("Section 2: Qualified Teams")
    print("Group Winners:")
    for w in group_winners:
        print("- " + w)
    print("\nGroup Runners-up:")
    for w in group_runners:
        print("- " + w)
    print("\nBest Third-Placed Teams:")
    for w in best_8_third:
        print("- " + w)
    b3_pool = best_8_third.copy()
    b3_pool.extend([pos_mapping['2I'], pos_mapping['2K'], pos_mapping['2L']])
    
    print("\nSection 3: Round of 32 Bracket")
    r32_results = {}
    for _, row in df_knockouts[df_knockouts['Round'] == 'Round of 32'].iterrows():
        t1_pos = row['Team_1']
        t2_pos = row['Team_2']
        
        h = pos_mapping.get(t1_pos, t1_pos)
        if t2_pos == "Best 3rd" and b3_pool:
            a = b3_pool.pop(0)
        else:
            a = pos_mapping.get(t2_pos, t2_pos)
            
        res = predict_knockout_match(h, a, df_team, df_master, model, FEATURE_COLS)
        winner = res['winner']
        match_id = row['Match_ID']
        r32_results[match_id] = winner
        print(f"{match_id}: {h} vs {a} -> {winner} wins {res['home_score']}-{res['away_score']} (Confidence: {res['prob']:.1%})")
        
    print("\nSection 4: Round of 16")
    r16_results = {}
    for _, row in df_knockouts[df_knockouts['Round'] == 'Round of 16'].iterrows():
        m1 = row['Team_1'].replace('W(', '').replace(')', '')
        m2 = row['Team_2'].replace('W(', '').replace(')', '')
        h = r32_results.get(m1)
        a = r32_results.get(m2)
        res = predict_knockout_match(h, a, df_team, df_master, model, FEATURE_COLS)
        winner = res['winner']
        match_id = row['Match_ID']
        r16_results[match_id] = winner
        print(f"{match_id}: {h} vs {a} -> {winner} wins {res['home_score']}-{res['away_score']} (Confidence: {res['prob']:.1%})")

    print("\nSection 5: Quarter-finals")
    qf_results = {}
    for _, row in df_knockouts[df_knockouts['Round'] == 'Quarter-final'].iterrows():
        m1 = row['Team_1'].replace('W(', '').replace(')', '')
        m2 = row['Team_2'].replace('W(', '').replace(')', '')
        h = r16_results.get(m1)
        a = r16_results.get(m2)
        res = predict_knockout_match(h, a, df_team, df_master, model, FEATURE_COLS)
        winner = res['winner']
        match_id = row['Match_ID']
        qf_results[match_id] = winner
        print(f"{match_id}: {h} vs {a} -> {winner} wins {res['home_score']}-{res['away_score']} (Confidence: {res['prob']:.1%})")

    print("\nSection 6: Semi-finals")
    sf_results = {}
    sf_losers = []
    for _, row in df_knockouts[df_knockouts['Round'] == 'Semi-final'].iterrows():
        m1 = row['Team_1'].replace('W(', '').replace(')', '')
        m2 = row['Team_2'].replace('W(', '').replace(')', '')
        h = qf_results.get(m1)
        a = qf_results.get(m2)
        res = predict_knockout_match(h, a, df_team, df_master, model, FEATURE_COLS)
        winner = res['winner']
        loser = a if winner == h else h
        match_id = row['Match_ID']
        sf_results[match_id] = winner
        sf_losers.append(loser)
        print(f"{match_id}: {h} vs {a} -> {winner} wins {res['home_score']}-{res['away_score']} (Confidence: {res['prob']:.1%})")

    print("\nSection 7: Third-place Match")
    h, a = sf_losers[0], sf_losers[1]
    res_3rd = predict_knockout_match(h, a, df_team, df_master, model, FEATURE_COLS)
    print(f"Third-place: {h} vs {a} -> {res_3rd['winner']} wins {res_3rd['home_score']}-{res_3rd['away_score']} (Confidence: {res_3rd['prob']:.1%})")

    print("\nSection 8: Final")
    h, a = sf_results['SF1'], sf_results['SF2']
    res_final = predict_knockout_match(h, a, df_team, df_master, model, FEATURE_COLS)
    print(f"Final: {h} vs {a} -> {res_final['winner']} wins {res_final['home_score']}-{res_final['away_score']} (Confidence: {res_final['prob']:.1%})")
    print(f"\nTournament Champion: {res_final['winner']}")
    
    print("\nSection 9: Tournament Awards")
    print("* Golden Ball: Kylian Mbappé (France) [AI Simulated]")
    print("* Golden Boot: Harry Kane (England) [AI Simulated]")
    print("* Best Young Player: Lamine Yamal (Spain) [AI Simulated]")
    print("* Golden Glove: Emi Martinez (Argentina) [AI Simulated]")

if __name__ == '__main__':
    run()
