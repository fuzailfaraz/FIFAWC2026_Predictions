"""
eloratings.net — Full Data Pipeline
====================================
Collects every international match from January 1, 2000 to today,
plus pre-match Elo ratings for both teams in every match.

HOW TO RUN
----------
    pip install requests pandas tqdm
    python elo_pipeline.py

OUTPUT FILES
------------
    matches_df.csv          — one row per match, 2000-present
    elo_ratings_df.csv      — one row per team, current Elo snapshot
    combined_df.csv         — matches + pre-match Elo for both teams (ready for ML)

HOW IT WORKS
------------
eloratings.net is a single-page app backed by plain TSV files.
No JavaScript execution needed — pure HTTPS GET requests.

  teams.tsv            → team code → successor mapping
  en.teams.tsv         → team code → English name
  tournaments.tsv      → tournament code → name list
  {TEAM}_results.tsv   → all match history for one team (pushMatchRow layout)
  World.tsv            → current global Elo ratings (pushRatingRow layout)

TSV SCHEMAS (reverse-engineered from eloratings.net/scripts/ratings.js)
------------------------------------------------------------------------
Match row (fields split on \\t):
  [0] year   [1] month  [2] day
  [3] team1_code  [4] team2_code
  [5] score1  [6] score2
  [7] tournament_code  [8] venue_code (or team1 if home)
  [9] elo_change (for team1)
  [10] rating1_after  [11] rating2_after
  [12] rank_move1     [13] rank_move2
  [14] rank1_after    [15] rank2_after

Ratings row (World.tsv):
  [0] local_rank  [1] global_rank  [2] team_code
  [3] rating
  [4] rank_max    [5] rating_max
  [6] rank_avg    [7] rating_avg
  [8] rank_min    [9] rating_min
  [10-21] rank/rating changes (3m,6m,1y,2y,5y,10y)
  [22] total_matches  [23] home  [24] away  [25] neutral
  [26] wins  [27] losses  [28] draws
  [29] goals_for  [30] goals_against
  [31] rank_chg   [32] rating_chg
"""

import requests
import pandas as pd
import numpy as np
import time
import sys
from datetime import datetime, date
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("tqdm not installed — progress will use simple counter. pip install tqdm for progress bars.")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL   = "https://www.eloratings.net/"
START_DATE = date(2000, 1, 1)       # collect matches from this date onward
END_DATE   = date.today()           # through today
DELAY      = 0.1                    # seconds between requests (be respectful)
MAX_RETRY  = 3                      # retries per failed request
OUT_DIR    = Path("data")           # where to save CSVs

HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept":          "text/plain, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Referer":         "https://www.eloratings.net/",
    "Origin":          "https://www.eloratings.net",
    "Sec-Fetch-Dest":  "empty",
    "Sec-Fetch-Mode":  "cors",
    "Sec-Fetch-Site":  "same-origin",
    "X-Requested-With":"XMLHttpRequest",
    "Connection":      "keep-alive",
}


# ─────────────────────────────────────────────────────────────────────────────
# HTTP HELPER
# ─────────────────────────────────────────────────────────────────────────────

session = requests.Session()
session.headers.update(HEADERS)

def fetch_tsv(endpoint: str, retry: int = 0) -> str | None:
    """
    Fetch a TSV file from eloratings.net.
    Returns raw text or None on permanent failure.
    """
    url = BASE_URL + endpoint
    try:
        r = session.get(url, timeout=30)
        if r.status_code == 200:
            return r.text
        elif r.status_code == 429:
            wait = 30 + retry * 30
            print(f"\n  Rate limited — waiting {wait}s before retry...")
            time.sleep(wait)
            if retry < MAX_RETRY:
                return fetch_tsv(endpoint, retry + 1)
            return None
        elif r.status_code == 404:
            return None          # team doesn't exist — normal
        else:
            if retry < MAX_RETRY:
                time.sleep(5 * (retry + 1))
                return fetch_tsv(endpoint, retry + 1)
            print(f"\n  FAILED {url} — HTTP {r.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        if retry < MAX_RETRY:
            print(f"\n  Retrying {url} ({retry+1}/{MAX_RETRY}) after error: {e}")
            time.sleep(5 * (retry + 1))
            return fetch_tsv(endpoint, retry + 1)
        print(f"\n  PERMANENT FAIL {url}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PART 0 — LOAD LOOKUP TABLES
# ─────────────────────────────────────────────────────────────────────────────

def load_team_codes() -> dict:
    """
    Load teams.tsv → {team_code: successor_code}
    and en.teams.tsv → {team_code: english_name}
    Returns combined dict: {code: {'name': str, 'successor': str|None}}
    """
    print("Loading team lookup tables...")

    # teams.tsv: successor mapping (team_code \\t successor_code)
    successor = {}
    raw = fetch_tsv("teams.tsv")
    if raw:
        for line in raw.strip().split('\n'):
            parts = line.split('\t')
            if len(parts) >= 2:
                successor[parts[0]] = parts[1]

    # en.teams.tsv: English names (code \\t name1 \\t name2 \\t ...)
    # The JS uses the last name in the list for the widest layout
    names = {}
    raw = fetch_tsv("en.teams.tsv")
    if raw:
        for line in raw.strip().split('\n'):
            parts = line.split('\t')
            if len(parts) >= 2:
                code = parts[0]
                # Skip _loc suffix entries (locative forms)
                if code.endswith('_loc'):
                    continue
                # Use the first (longest) name
                names[code] = parts[1]

    print(f"  Loaded {len(names)} teams, {len(successor)} successor mappings")

    teams = {}
    for code, name in names.items():
        teams[code] = {
            'name': name,
            'successor': successor.get(code)
        }
    return teams


def load_tournament_names() -> dict:
    """
    tournaments.tsv: {tournament_code: [name1, name2, ...]}
    Returns {code: primary_name}
    """
    print("Loading tournament lookup table...")
    tournaments = {}
    raw = fetch_tsv("en.tournaments.tsv")
    if raw:
        for line in raw.strip().split('\n'):
            parts = line.split('\t')
            if len(parts) >= 2:
                tournaments[parts[0]] = parts[1]
    print(f"  Loaded {len(tournaments)} tournament names")
    return tournaments


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — COLLECT MATCH RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def parse_match_row(line: str, teams: dict, tournaments: dict) -> dict | None:
    """
    Parse one line from {TEAM}_results.tsv using the pushMatchRow schema.

    TSV columns (tab-separated):
      0: year   1: month   2: day
      3: team1_code   4: team2_code
      5: score1   6: score2
      7: tournament_code   8: venue_code
      9: elo_change (positive = team1 gained)
      10: rating1_after   11: rating2_after
      12: rank_move1      13: rank_move2
      14: rank1_after     15: rank2_after
    """
    fields = line.rstrip('\n').split('\t')
    if len(fields) < 12:
        return None

    try:
        year  = int(fields[0])
        month = int(fields[1])
        day   = int(fields[2])
        match_date = date(year, month, day)
    except (ValueError, IndexError):
        return None

    # Filter to date range
    if match_date < START_DATE or match_date > END_DATE:
        return None

    team1_code = fields[3]
    team2_code = fields[4]

    try:
        home_score = int(fields[5])
        away_score = int(fields[6])
    except (ValueError, IndexError):
        return None

    tournament_code = fields[7] if len(fields) > 7 else ''
    venue_code      = fields[8] if len(fields) > 8 else ''

    try:
        elo_change   = float(fields[9])   if len(fields) > 9  else None
        rating1_after = float(fields[10]) if len(fields) > 10 else None
        rating2_after = float(fields[11]) if len(fields) > 11 else None
        rank1_after   = int(fields[14])   if len(fields) > 14 else None
        rank2_after   = int(fields[15])   if len(fields) > 15 else None
    except (ValueError, IndexError):
        elo_change = rating1_after = rating2_after = rank1_after = rank2_after = None

    # Resolve team names
    t1_name = teams.get(team1_code, {}).get('name', team1_code)
    t2_name = teams.get(team2_code, {}).get('name', team2_code)

    # Neutral venue: if venue_code != team1_code, it's neutral
    # (venue_code is the host team, or empty for proper neutral)
    neutral = (venue_code != team1_code) if venue_code else False

    tournament_name = tournaments.get(tournament_code, tournament_code)

    # Derive Elo BEFORE match from Elo AFTER match
    # elo_change is the delta for team1 (team2 gets -elo_change)
    rating1_before = (rating1_after - elo_change) if (rating1_after is not None and elo_change is not None) else None
    rating2_before = (rating2_after + elo_change) if (rating2_after is not None and elo_change is not None) else None

    return {
        'date':              str(match_date),
        'year':              year,
        'month':             month,
        'day':               day,
        'home_team_code':    team1_code,
        'away_team_code':    team2_code,
        'home_team':         t1_name,
        'away_team':         t2_name,
        'home_score':        home_score,
        'away_score':        away_score,
        'result':            'H' if home_score > away_score else ('A' if home_score < away_score else 'D'),
        'total_goals':       home_score + away_score,
        'tournament_code':   tournament_code,
        'tournament':        tournament_name,
        'neutral_venue':     neutral,
        'venue_code':        venue_code,
        'elo_change':        elo_change,          # team1's change (team2 gets negative)
        # AFTER match ratings (useful for building rolling time series)
        'home_elo_after':    rating1_after,
        'away_elo_after':    rating2_after,
        'home_rank_after':   rank1_after,
        'away_rank_after':   rank2_after,
        # BEFORE match ratings (key feature for ML — no leakage)
        'home_elo_before':   rating1_before,
        'away_elo_before':   rating2_before,
        'elo_diff_before':   (rating1_before - rating2_before) if (rating1_before and rating2_before) else None,
    }


def fetch_team_matches(team_code: str, teams: dict, tournaments: dict) -> list[dict]:
    """Fetch and parse all results for one team."""
    raw = fetch_tsv(f"{team_code}_results.tsv")
    if not raw:
        return []

    rows = []
    for line in raw.strip().split('\n'):
        if not line.strip():
            continue
        row = parse_match_row(line, teams, tournaments)
        if row:
            rows.append(row)
    return rows


def collect_all_matches(teams: dict, tournaments: dict) -> pd.DataFrame:
    """
    Fetch match history for every known team.
    Deduplicates so each match appears only once.
    Returns DataFrame sorted by date.
    """
    print(f"\nCollecting matches from {START_DATE} to {END_DATE}...")
    print(f"  Total teams to query: {len(teams)}")

    all_rows = []
    seen_matches = set()   # dedup key: (date, sorted team pair, score)
    failed_teams = []

    team_list = list(teams.keys())

    if HAS_TQDM:
        iterator = tqdm(team_list, desc="Teams", unit="team")
    else:
        iterator = team_list

    for i, code in enumerate(iterator):
        if not HAS_TQDM and i % 25 == 0:
            print(f"  Progress: {i}/{len(team_list)} teams ({len(all_rows)} matches so far)")

        rows = fetch_team_matches(code, teams, tournaments)

        for row in rows:
            # Dedup key: canonical ordering of teams doesn't matter, date + both teams + score
            key = (
                row['date'],
                tuple(sorted([row['home_team_code'], row['away_team_code']])),
                row['home_score'],
                row['away_score']
            )
            if key not in seen_matches:
                seen_matches.add(key)
                all_rows.append(row)

        time.sleep(DELAY)

    if not all_rows:
        print("WARNING: No matches collected. Check network connectivity.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"\n  ✓ Collected {len(df):,} unique matches across {df['home_team'].nunique()} teams")
    print(f"    Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"    Tournaments: {df['tournament'].nunique()} unique")
    print(f"    Null elo_before rate: {df['home_elo_before'].isna().mean():.1%}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — COLLECT CURRENT ELO RATINGS
# ─────────────────────────────────────────────────────────────────────────────

def parse_ratings_row(line: str, teams: dict) -> dict | None:
    """
    Parse one line from World.tsv using the pushRatingRow schema.

    Columns:
      0: local_rank  1: global_rank  2: team_code  3: rating
      4: rank_max    5: rating_max
      6: rank_avg    7: rating_avg
      8: rank_min    9: rating_min
      10: rank_3m_chg   11: rating_3m_chg
      12: rank_6m_chg   13: rating_6m_chg
      14: rank_1y_chg   15: rating_1y_chg
      16: rank_2y_chg   17: rating_2y_chg
      18: rank_5y_chg   19: rating_5y_chg
      20: rank_10y_chg  21: rating_10y_chg
      22: total_matches 23: home_matches 24: away_matches 25: neutral_matches
      26: wins  27: losses  28: draws
      29: goals_for  30: goals_against
      31: rank_change  32: rating_change
    """
    fields = line.rstrip('\n').split('\t')
    if len(fields) < 10:
        return None

    team_code = fields[2] if len(fields) > 2 else None
    if not team_code:
        return None

    team_name = teams.get(team_code, {}).get('name', team_code)

    def safe_int(idx, default=None):
        try: return int(fields[idx])
        except: return default

    def safe_float(idx, default=None):
        try: return float(fields[idx])
        except: return default

    return {
        'team_code':       team_code,
        'team':            team_name,
        'global_rank':     safe_int(1),
        'elo_rating':      safe_float(3),
        'rank_max':        safe_int(4),
        'rating_max':      safe_float(5),
        'rank_avg':        safe_int(6),
        'rating_avg':      safe_float(7),
        'rank_min':        safe_int(8),
        'rating_min':      safe_float(9),
        'rank_chg_3m':     safe_int(10),
        'rating_chg_3m':   safe_float(11),
        'rank_chg_6m':     safe_int(12),
        'rating_chg_6m':   safe_float(13),
        'rank_chg_1y':     safe_int(14),
        'rating_chg_1y':   safe_float(15),
        'rank_chg_2y':     safe_int(16),
        'rating_chg_2y':   safe_float(17),
        'rank_chg_5y':     safe_int(18),
        'rating_chg_5y':   safe_float(19),
        'rank_chg_10y':    safe_int(20),
        'rating_chg_10y':  safe_float(21),
        'total_matches':   safe_int(22),
        'home_matches':    safe_int(23),
        'away_matches':    safe_int(24),
        'neutral_matches': safe_int(25),
        'total_wins':      safe_int(26),
        'total_losses':    safe_int(27),
        'total_draws':     safe_int(28),
        'goals_for':       safe_int(29),
        'goals_against':   safe_int(30),
        'rank_change':     safe_int(31),
        'rating_change':   safe_float(32),
    }


def collect_elo_ratings(teams: dict) -> pd.DataFrame:
    """
    Download World.tsv — the global Elo ratings snapshot.
    Returns DataFrame with one row per team (current ratings).
    """
    print("\nDownloading current Elo ratings (World.tsv)...")
    raw = fetch_tsv("World.tsv")
    if not raw:
        print("  WARNING: Could not fetch World.tsv")
        return pd.DataFrame()

    rows = []
    for line in raw.strip().split('\n'):
        if not line.strip():
            continue
        row = parse_ratings_row(line, teams)
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values('global_rank').reset_index(drop=True)

    print(f"  ✓ Loaded {len(df)} teams from World.tsv")
    if not df.empty:
        print(f"    Rating range: {df['elo_rating'].min():.0f} – {df['elo_rating'].max():.0f}")
        top5 = df.head(5)[['global_rank','team','elo_rating']].to_string(index=False)
        print(f"    Top 5:\n{top5}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — BUILD COMBINED DATASET WITH PRE-MATCH ELO
# ─────────────────────────────────────────────────────────────────────────────

def build_pre_match_elo_from_history(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    The {TEAM}_results.tsv rows include:
      - rating_after: Elo AFTER this match
      - elo_change:   how much team1's Elo changed

    So: rating_before = rating_after - elo_change  (for team1)
        rating_before = rating_after + elo_change  (for team2)

    This is already computed in parse_match_row() as home_elo_before / away_elo_before.

    HOWEVER: for matches where elo_before is null (first appearance of a team
    in a match, or scraping gaps), we fall back to a per-team rolling
    Elo time series reconstructed from all available matches.

    This function fills those gaps.
    """
    if matches_df.empty:
        return matches_df

    df = matches_df.copy().sort_values('date')

    # How many have pre-match Elo already computed?
    n_total    = len(df)
    n_have_elo = df['home_elo_before'].notna().sum()
    print(f"\n  Pre-match Elo coverage: {n_have_elo:,}/{n_total:,} ({n_have_elo/n_total:.1%})")

    if n_have_elo == n_total:
        print("  All rows have pre-match Elo — no gap-filling needed.")
        return df

    # For rows missing pre-match Elo, reconstruct from rolling history
    print("  Gap-filling missing pre-match Elo from rolling time series...")

    # Build a per-team rolling Elo dictionary: team_code → elo_value
    # We process chronologically; whenever we see a match, we update both teams' Elo
    team_elo_current = {}   # {team_code: current_elo}

    for idx, row in df.iterrows():
        h = row['home_team_code']
        a = row['away_team_code']

        # Fill if missing
        if pd.isna(row['home_elo_before']) and h in team_elo_current:
            df.at[idx, 'home_elo_before'] = team_elo_current[h]
        if pd.isna(row['away_elo_before']) and a in team_elo_current:
            df.at[idx, 'away_elo_before'] = team_elo_current[a]

        # Update rolling tracker with post-match ratings
        if pd.notna(row['home_elo_after']):
            team_elo_current[h] = row['home_elo_after']
        if pd.notna(row['away_elo_after']):
            team_elo_current[a] = row['away_elo_after']

    # Recompute elo_diff_before
    df['elo_diff_before'] = df['home_elo_before'] - df['away_elo_before']

    n_filled = df['home_elo_before'].notna().sum()
    print(f"  After gap-fill: {n_filled:,}/{n_total:,} ({n_filled/n_total:.1%})")

    return df


def build_combined_dataset(matches_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge matches with current Elo ratings to add additional team stats.
    The pre-match Elo (home_elo_before / away_elo_before) is already in matches_df.
    This adds current rating stats (win/loss record, etc.) for context.
    """
    if matches_df.empty:
        return matches_df

    combined = matches_df.copy()

    # Add current team stats from ratings snapshot (for the team profiles section)
    if not ratings_df.empty:
        home_stats = ratings_df[['team_code','total_wins','total_losses','total_draws',
                                  'goals_for','goals_against','rating_chg_1y']].copy()
        home_stats.columns = ['home_team_code','home_total_wins','home_total_losses',
                               'home_total_draws','home_goals_for','home_goals_against',
                               'home_rating_chg_1y']
        away_stats = ratings_df[['team_code','total_wins','total_losses','total_draws',
                                  'goals_for','goals_against','rating_chg_1y']].copy()
        away_stats.columns = ['away_team_code','away_total_wins','away_total_losses',
                               'away_total_draws','away_goals_for','away_goals_against',
                               'away_rating_chg_1y']
        combined = combined.merge(home_stats, on='home_team_code', how='left')
        combined = combined.merge(away_stats, on='away_team_code', how='left')

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  eloratings.net — International Football Match Data Pipeline")
    print(f"  Date range: {START_DATE} → {END_DATE}")
    print("=" * 70)

    OUT_DIR.mkdir(exist_ok=True)

    # ── Step 0: Load lookup tables ────────────────────────────────────────
    teams       = load_team_codes()
    tournaments = load_tournament_names()
    time.sleep(DELAY)

    # ── Step 1: Collect all match results ─────────────────────────────────
    matches_df = collect_all_matches(teams, tournaments)

    if not matches_df.empty:
        # Fill any gaps in pre-match Elo
        matches_df = build_pre_match_elo_from_history(matches_df)

        out_path = OUT_DIR / "matches_df.csv"
        matches_df.to_csv(out_path, index=False)
        print(f"\n  ✓ Saved → {out_path}  ({len(matches_df):,} rows × {len(matches_df.columns)} cols)")

    # ── Step 2: Collect current Elo ratings ───────────────────────────────
    time.sleep(DELAY)
    ratings_df = collect_elo_ratings(teams)

    if not ratings_df.empty:
        out_path = OUT_DIR / "elo_ratings_df.csv"
        ratings_df.to_csv(out_path, index=False)
        print(f"  ✓ Saved → {out_path}  ({len(ratings_df):,} rows × {len(ratings_df.columns)} cols)")

    # ── Step 3: Build combined dataset ────────────────────────────────────
    if not matches_df.empty:
        combined_df = build_combined_dataset(matches_df, ratings_df)
        out_path = OUT_DIR / "combined_df.csv"
        combined_df.to_csv(out_path, index=False)
        print(f"  ✓ Saved → {out_path}  ({len(combined_df):,} rows × {len(combined_df.columns)} cols)")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)

    if not matches_df.empty:
        print(f"\n  matches_df.csv")
        print(f"    Rows:       {len(matches_df):,} matches")
        print(f"    Date range: {matches_df['date'].min().date()} → {matches_df['date'].max().date()}")
        print(f"    Teams:      {matches_df['home_team'].nunique()} unique national teams")
        print(f"    Tournaments:{matches_df['tournament'].nunique()} unique tournaments")
        print(f"    Columns:    {matches_df.columns.tolist()}")

        print(f"\n  Pre-match Elo coverage:")
        print(f"    home_elo_before: {matches_df['home_elo_before'].notna().sum():,}/{len(matches_df):,}")
        print(f"    away_elo_before: {matches_df['away_elo_before'].notna().sum():,}/{len(matches_df):,}")

    if not ratings_df.empty:
        print(f"\n  elo_ratings_df.csv")
        print(f"    Teams: {len(ratings_df)}")
        print(f"    Top 10 by Elo:")
        top10 = ratings_df.head(10)[['global_rank','team','elo_rating','rating_chg_1y']]
        print(top10.to_string(index=False))

    print("\n  Column reference for matches_df.csv:")
    col_docs = {
        'date':             'Match date (YYYY-MM-DD)',
        'home_team':        'Home team name (from eloratings.net)',
        'away_team':        'Away team name',
        'home_score':       'Goals scored by home team',
        'away_score':       'Goals scored by away team',
        'result':           'H=home win, D=draw, A=away win  ← YOUR ML LABEL',
        'tournament':       'Competition name',
        'neutral_venue':    'True if match at neutral ground',
        'home_elo_before':  'Home team Elo BEFORE match  ← KEY FEATURE (no leakage)',
        'away_elo_before':  'Away team Elo BEFORE match  ← KEY FEATURE (no leakage)',
        'elo_diff_before':  'home_elo_before - away_elo_before',
        'home_elo_after':   'Home Elo after match (for building rolling series)',
        'away_elo_after':   'Away Elo after match',
        'elo_change':       'Elo points gained by home team (+ve = home gained)',
    }
    for col, desc in col_docs.items():
        print(f"    {col:<22} {desc}")


if __name__ == '__main__':
    main()
