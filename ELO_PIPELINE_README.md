# eloratings.net — Full Data Pipeline

## What you get

| File | Contents | Rows |
|------|----------|------|
| `matches_df.csv` | Every international match Jan 2000–today | ~10,000+ |
| `elo_ratings_df.csv` | Current Elo snapshot for all ~250 teams | ~250 |
| `combined_df.csv` | matches + team context stats merged | same as matches |

---

## Setup (run once)

```bash
pip install requests pandas tqdm
python elo_pipeline.py
```

Expected runtime: **25–40 minutes** (≈210 teams × 0.8s delay = ~170s + download time).
No API key needed. No login. Plain HTTPS GET requests.

---

## Key columns in `matches_df.csv`

| Column | Use |
|--------|-----|
| `date` | Match date |
| `home_team` / `away_team` | Team names as on eloratings.net |
| `home_score` / `away_score` | Goals |
| `result` | **H / D / A** ← your ML label |
| `tournament` | Competition name |
| `neutral_venue` | True/False |
| `home_elo_before` | Home Elo BEFORE match ← **no data leakage** |
| `away_elo_before` | Away Elo BEFORE match ← **no data leakage** |
| `elo_diff_before` | home_elo_before − away_elo_before |
| `elo_change` | Elo points home team gained (away gets −elo_change) |
| `home_elo_after` / `away_elo_after` | Post-match Elo (for building time series) |

---

## How pre-match Elo is computed (zero leakage)

Each row in `{TEAM}_results.tsv` stores:
- `rating_after` — Elo after this match
- `elo_change` — how much the home team's Elo changed

So: `rating_before = rating_after − elo_change`

This is computed directly from the source data — no approximation.
For any remaining gaps (e.g. first appearance of a new team), the pipeline
fills using a rolling chronological time series from prior matches.

---

## How to use in your WC 2026 project

```python
import pandas as pd

df = pd.read_csv('matches_df.csv', parse_dates=['date'])

# Filter to relevant time period
df_recent = df[df['date'].dt.year >= 2010]

# Key ML features (all pre-match, no leakage)
features = [
    'home_elo_before',
    'away_elo_before',
    'elo_diff_before',
    'neutral_venue',
]

# Label
label = 'result'   # 'H', 'D', 'A'

# Filter to meaningful competitions
competitive = df_recent[~df_recent['tournament'].str.contains('Friendly', na=False)]
```

---

## eloratings.net data structure (for reference)

The site is a single-page app backed by plain `.tsv` files.
No JavaScript execution needed — pure HTTP GET.

### Endpoints

| URL | Content |
|-----|---------|
| `eloratings.net/teams.tsv` | Team code → successor mapping |
| `eloratings.net/en.teams.tsv` | Team code → English name |
| `eloratings.net/en.tournaments.tsv` | Tournament code → name |
| `eloratings.net/World.tsv` | Current global ratings (all teams) |
| `eloratings.net/{TEAM}_results.tsv` | All match history for one team |
| `eloratings.net/{TEAM}.tsv` | Rating history page for one team |
| `eloratings.net/latest.tsv` | Most recent matches globally |
| `eloratings.net/fixtures.tsv` | Upcoming fixtures |

### Match TSV column layout (from `ratings.js` → `pushMatchRow`)

```
col  0: year
col  1: month
col  2: day
col  3: home_team_code    (team1 in source)
col  4: away_team_code    (team2 in source)
col  5: home_score
col  6: away_score
col  7: tournament_code
col  8: venue_code        (home team = neutral if != col 3)
col  9: elo_change        (positive = home team gained points)
col 10: home_elo_after
col 11: away_elo_after
col 12: home_rank_move
col 13: away_rank_move
col 14: home_rank_after
col 15: away_rank_after
```

### Ratings TSV column layout (from `ratings.js` → `pushRatingRow`)

```
col  0: local_rank
col  1: global_rank
col  2: team_code
col  3: current_elo_rating
col  4-9:  rank/rating max, avg, min
col 10-21: rank/rating changes (3m, 6m, 1y, 2y, 5y, 10y)
col 22-25: match counts (total, home, away, neutral)
col 26-28: wins, losses, draws
col 29-30: goals_for, goals_against
col 31-32: rank_change, rating_change (since last update)
```

---

## Notes

- The pipeline requests each team's file with a 0.8s delay — adjust `DELAY` in the script if needed
- If interrupted, you can re-run — existing data won't be lost (just re-fetched)
- Walkover and forfeit results are excluded upstream by eloratings.net
- Data is released by eloratings.net under a permissive reuse policy
