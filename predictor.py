import pandas as pd
import numpy as np
import pickle
import shap

def predict_match(home_team: str, away_team: str,
                  df_master: pd.DataFrame, model, explainer,
                  feature_cols: list) -> dict:
    """
    Look up the pre-computed row for this fixture in wc2026_master_dataset.csv,
    run the model, return probabilities and SHAP values.
    """
    row = df_master[
        (df_master['home_team'] == home_team) &
        (df_master['away_team'] == away_team)
    ]
    if row.empty:
        # Check reverse just in case, though 2026 group stage is fixed
        row = df_master[
            (df_master['home_team'] == away_team) &
            (df_master['away_team'] == home_team)
        ]
        if row.empty:
            return None

    X_pred     = row[feature_cols]
    
    # Run the model
    proba      = model.predict_proba(X_pred)[0]
    
    # Label mapping
    # XGBoost trained with LabelEncoder has classes [0, 1, 2] corresponding to ['A', 'D', 'H'] alphabetically.
    # If the model has classes_ attribute, we use it, otherwise use ['A', 'D', 'H']
    if hasattr(model, 'classes_'):
        classes = list(model.classes_)
    else:
        classes = ['A', 'D', 'H']
        
    # Standardize classes to string codes H, D, A
    # In case model.classes_ are integers 0, 1, 2, we map them
    decoded_classes = []
    for c in classes:
        if c == 0 or c == '0':
            decoded_classes.append('A')
        elif c == 1 or c == '1':
            decoded_classes.append('D')
        elif c == 2 or c == '2':
            decoded_classes.append('H')
        else:
            decoded_classes.append(str(c))
            
    prob_dict  = dict(zip(decoded_classes, proba))
    
    # Calculate SHAP values
    sv = explainer.shap_values(X_pred)
    
    # --- ELO RATINGS BLENDING ---
    try:
        elo_df = pd.read_csv('data/elo_ratings_df.csv')
        h_elo = elo_df[elo_df['team'] == home_team]['elo_rating'].values[0] if home_team in elo_df['team'].values else 1500
        a_elo = elo_df[elo_df['team'] == away_team]['elo_rating'].values[0] if away_team in elo_df['team'].values else 1500
    except:
        h_elo = 1500; a_elo = 1500
        
    elo_diff = h_elo - a_elo
    prob_h_elo = 1 / (1 + 10 ** (-elo_diff / 400))
    prob_a_elo = 1 - prob_h_elo
    
    old_h = prob_dict.get('H', 0.0)
    old_a = prob_dict.get('A', 0.0)
    old_d = prob_dict.get('D', 0.0)
    
    # 70% weight to Elo for realistic recent form
    new_h = old_h * 0.3 + prob_h_elo * 0.7
    new_a = old_a * 0.3 + prob_a_elo * 0.7
    new_d = old_d * 0.3 + 0.05
    
    total = new_h + new_a + new_d
    prob_dict['H'] = new_h / total
    prob_dict['A'] = new_a / total
    prob_dict['D'] = new_d / total
    
    if prob_dict['H'] > prob_dict['A'] and prob_dict['H'] > prob_dict['D']:
        pred_class_decoded = 'H'
    elif prob_dict['A'] > prob_dict['H'] and prob_dict['A'] > prob_dict['D']:
        pred_class_decoded = 'A'
    else:
        pred_class_decoded = 'D'
        
    cls_idx = decoded_classes.index(pred_class_decoded)
        
    # Extract shap values for the predicted class
    if isinstance(sv, list):
        shap_row = sv[cls_idx][0]
    elif isinstance(sv, np.ndarray):
        if len(sv.shape) == 3:
            if sv.shape[0] == len(classes): shap_row = sv[cls_idx][0]
            elif sv.shape[2] == len(classes): shap_row = sv[0, :, cls_idx]
            else: shap_row = sv[0, cls_idx, :]
        elif len(sv.shape) == 2:
            shap_row = sv[0]
        else:
            shap_row = sv
    else:
        shap_row = sv
        
    # Get base value (expected value) for predicted class
    try:
        base_val_raw = explainer.expected_value[cls_idx]
    except (TypeError, IndexError, KeyError):
        base_val_raw = explainer.expected_value
        
    # Ensure base_value is a scalar float
    if isinstance(base_val_raw, (list, np.ndarray)):
        base_value = float(base_val_raw[0])
    else:
        base_value = float(base_val_raw)
        
    return {
        'home_win': prob_dict.get('H', 0.0),
        'draw':     prob_dict.get('D', 0.0),
        'away_win': prob_dict.get('A', 0.0),
        'prediction': pred_class_decoded,
        'shap_values': shap_row,
        'feature_names': feature_cols,
        'base_value': base_value,
        'fixture_row': row
    }
