import os
import pandas as pd
import requests
from datetime import datetime

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"
DATA_PATH = "data/raw/matches.parquet"
TEAMS_PATH = "data/raw/teams.parquet"

LEAGUES = [
    "eng.1",
    "esp.1",
    "ger.1",
    "ita.1",
    "fra.1",
    "uefa.champions",
    "fifa.friendlies",
    "fifa.international",
    "fifa.friendly"
]

HEADERS = {"User-Agent": "Mozilla/5.0"}

def fetch_json(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None

def ensure_dirs():
    os.makedirs("data/raw", exist_ok=True)

def load_existing():
    if os.path.exists(DATA_PATH):
        return pd.read_parquet(DATA_PATH)
    return pd.DataFrame()

def save(df):
    df.drop_duplicates(subset=["event_id"], inplace=True)
    df.to_parquet(DATA_PATH, index=False)

def parse_scoreboard(league):
    url = f"{BASE_URL}/{league}/scoreboard"
    data = fetch_json(url)
    if not data:
        return []

    events = data.get("events", [])
    rows = []

    for e in events:
        try:
            comp = data.get("leagues", [{}])[0].get("name")

            comp_type = "club"
            if "fifa" in league:
                comp_type = "international"

            competitors = e.get("competitions", [{}])[0].get("competitors", [])

            home = None
            away = None

            for c in competitors:
                if c.get("homeAway") == "home":
                    home = c
                else:
                    away = c

            rows.append({
                "event_id": e.get("id"),
                "date": e.get("date"),
                "competition": comp,
                "league": league,
                "competition_type": comp_type,
                "home_team": home.get("team", {}).get("displayName") if home else None,
                "away_team": away.get("team", {}).get("displayName") if away else None,
                "home_score": int(home.get("score", 0)) if home else None,
                "away_score": int(away.get("score", 0)) if away else None,
                "status": e.get("status", {}).get("type", {}).get("state"),
                "neutral_site": e.get("competitions", [{}])[0].get("neutralSite", False)
            })
        except:
            continue

    return rows

def main():
    ensure_dirs()

    existing = load_existing()
    existing_ids = set(existing["event_id"].astype(str).tolist()) if not existing.empty else set()

    new_rows = []

    for league in LEAGUES:
        rows = parse_scoreboard(league)
        for r in rows:
            if r["event_id"] and str(r["event_id"]) not in existing_ids:
                new_rows.append(r)

    if len(new_rows) == 0:
        return

    df_new = pd.DataFrame(new_rows)

    if not existing.empty:
        df = pd.concat([existing, df_new], ignore_index=True)
    else:
        df = df_new

    save(df)

if __name__ == "__main__":
    main()
