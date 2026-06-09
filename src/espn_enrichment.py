import pandas as pd
import requests
import time
import os
from datetime import datetime

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"

INPUT_PATH = "data/raw/matches.parquet"
OUTPUT_PATH = "data/processed/matches_enriched.parquet"

HEADERS = {"User-Agent": "Mozilla/5.0"}

def safe_get(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def extract_team_stats(summary_json):
    """
    Extrae estadísticas de boxscore ESPN si están disponibles
    """
    stats = {}

    try:
        box = summary_json.get("boxscore", {}).get("teams", [])

        for team in box:
            side = team.get("homeAway")
            stats[f"{side}_score"] = team.get("score")

            for s in team.get("statistics", []):
                label = s.get("label", "").lower()
                value = s.get("value", 0)

                if "shot" in label and "on target" in label:
                    stats[f"{side}_shots_on_target"] = value
                elif "shot" in label:
                    stats[f"{side}_shots"] = value
                elif "possession" in label:
                    stats[f"{side}_possession"] = value
                elif "yellow" in label:
                    stats[f"{side}_yellow_cards"] = value
                elif "red" in label:
                    stats[f"{side}_red_cards"] = value
                elif "corner" in label:
                    stats[f"{side}_corners"] = value

    except Exception:
        pass

    return stats


def enrich_match(row, league):
    event_id = row["event_id"]

    url = f"{BASE_URL}/{league}/summary?event={event_id}"
    data = safe_get(url)

    if not data:
        return None

    stats = extract_team_stats(data)

    enriched = {
        "event_id": event_id,
        "date": row.get("date"),
        "competition": row.get("competition"),
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "home_score": row.get("home_score"),
        "away_score": row.get("away_score"),
        "neutral_site": row.get("neutral_site", False),
        **stats
    }

    return enriched


def load_matches():
    return pd.read_parquet(INPUT_PATH)


def save_enriched(df):
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)


def run():
    df = load_matches()

    enriched_rows = []

    leagues = [
        "eng.1",
        "esp.1",
        "ger.1",
        "ita.1",
        "fra.1",
        "fifa.friendly",
        "fifa.international",
        "fifa.friendlies"
    ]

    seen = set()

    for _, row in df.iterrows():

        if row["event_id"] in seen:
            continue

        league = None
        for l in leagues:
            league = l
            break

        enriched = enrich_match(row, league)

        if enriched:
            enriched_rows.append(enriched)

        seen.add(row["event_id"])
        time.sleep(0.3)

    if not enriched_rows:
        print("No enrichment data available.")
        return

    out_df = pd.DataFrame(enriched_rows)

    save_enriched(out_df)

    print(f"Enriched dataset saved: {len(out_df)} matches")


if __name__ == "__main__":
    run()
