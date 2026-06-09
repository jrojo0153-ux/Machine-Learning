import pandas as pd
import numpy as np
import os

INPUT_PATH = "data/raw/match_details.parquet"
OUTPUT_PATH = "data/processed/features.parquet"

def load_data():
    df = pd.read_parquet(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df

def compute_team_history(df, team, date, window=5):
    past = df[
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < date)
    ].sort_values("date").tail(window)

    if past.empty:
        return {}

    goals_scored = []
    goals_conceded = []
    shots = []
    shots_on_target = []
    wins = []
    btts = []

    for _, row in past.iterrows():
        is_home = row["home_team"] == team

        gf = row["home_score"] if is_home else row["away_score"]
        ga = row["away_score"] if is_home else row["home_score"]

        goals_scored.append(gf)
        goals_conceded.append(ga)

        shots.append(row.get("home_shots" if is_home else "away_shots", 0))
        shots_on_target.append(row.get("home_shots_on_target" if is_home else "away_shots_on_target", 0))

        win = (gf > ga)
        wins.append(1 if win else 0)

        btts.append(1 if (row["home_score"] > 0 and row["away_score"] > 0) else 0)

    return {
        "goals_avg": np.mean(goals_scored),
        "conceded_avg": np.mean(goals_conceded),
        "shots_avg": np.mean(shots),
        "shots_on_target_avg": np.mean(shots_on_target),
        "win_rate": np.mean(wins),
        "btts_rate": np.mean(btts)
    }

def build_dataset(df):
    rows = []

    for _, row in df.iterrows():
        date = row["date"]

        home = row["home_team"]
        away = row["away_team"]

        home_stats = compute_team_history(df, home, date)
        away_stats = compute_team_history(df, away, date)

        if not home_stats or not away_stats:
            continue

        rows.append({
            "event_id": row["event_id"],
            "date": date,

            "team_home": home,
            "team_away": away,

            "home_goals_avg_last5": home_stats["goals_avg"],
            "away_goals_avg_last5": away_stats["goals_avg"],

            "home_conceded_avg_last5": home_stats["conceded_avg"],
            "away_conceded_avg_last5": away_stats["conceded_avg"],

            "home_shots_avg_last5": home_stats["shots_avg"],
            "away_shots_avg_last5": away_stats["shots_avg"],

            "home_shots_on_target_avg_last5": home_stats["shots_on_target_avg"],
            "away_shots_on_target_avg_last5": away_stats["shots_on_target_avg"],

            "home_win_rate_last10": home_stats["win_rate"],
            "away_win_rate_last10": away_stats["win_rate"],

            "home_btts_rate_last10": home_stats["btts_rate"],
            "away_btts_rate_last10": away_stats["btts_rate"],

            "label_home_win": int(row["home_score"] > row["away_score"]),
            "label_btts": int(row["home_score"] > 0 and row["away_score"] > 0),
            "label_total_goals": row["home_score"] + row["away_score"]
        })

    return pd.DataFrame(rows)

def main():
    df = load_data()
    features = build_dataset(df)

    os.makedirs("data/processed", exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)

    print(f"Features generated: {len(features)} rows")

if __name__ == "__main__":
    main()
