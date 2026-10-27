#!/usr/bin/env python3
"""
Plan C: build the same normalized CSVs from GitHub instead of any NBA host.

Both stats.nba.com and cdn.nba.com are NBA-controlled and are commonly blocked by
campus and corporate networks. This script touches neither. It downloads two
parquet files published by the sportsdataverse project as GitHub release assets,
sourced from ESPN, and reshapes them into the identical seven-table schema.

Two HTTP requests to github.com, about 750 KB, a few seconds. No API keys, no
rate limits, no bot detection. If your students can reach GitHub -- and they can,
or pip would not work -- this path works for all of them simultaneously.

Verified against the real files: the 2025-26 season covers 2025-10-21 through
2026-06-13, 1,326 games (1,235 regular season, 6 play-in, 85 playoff),
34,883 player-game rows, 591 players.

    pip install pandas pyarrow requests
    python nba_pull_espn.py --season 2026 --outdir ./nba_csv

NOTE ON --season: ESPN labels a season by its ENDING year. The 2025-26 NBA
season is season 2026. This is a genuinely good thing to point out in class --
it is a real-world example of why you document the semantics of a key column
instead of assuming everyone reads '2026' the same way.
"""

from __future__ import annotations

import argparse
import io
import os
import sys

import pandas as pd
import requests

BASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
PLAYER_URL = BASE + "/espn_nba_player_boxscores/player_box_{season}.parquet"
TEAM_URL = BASE + "/espn_nba_team_boxscores/team_box_{season}.parquet"

# ESPN encodes season type as an integer.
SEASON_TYPE = {2: "Regular Season", 3: "Playoffs", 5: "PlayIn"}

# ESPN column -> our column. Same target schema as nba_pull.py.
PLAYER_STATS = {
    "minutes": "min",
    "field_goals_made": "fgm", "field_goals_attempted": "fga",
    "three_point_field_goals_made": "fg3m", "three_point_field_goals_attempted": "fg3a",
    "free_throws_made": "ftm", "free_throws_attempted": "fta",
    "offensive_rebounds": "oreb", "defensive_rebounds": "dreb", "rebounds": "reb",
    "assists": "ast", "steals": "stl", "blocks": "blk", "turnovers": "tov",
    "fouls": "pf", "points": "pts", "plus_minus": "plus_minus",
}
TEAM_STATS = {
    "field_goals_made": "fgm", "field_goals_attempted": "fga",
    "three_point_field_goals_made": "fg3m", "three_point_field_goals_attempted": "fg3a",
    "free_throws_made": "ftm", "free_throws_attempted": "fta",
    "offensive_rebounds": "oreb", "defensive_rebounds": "dreb", "total_rebounds": "reb",
    "assists": "ast", "steals": "stl", "blocks": "blk", "total_turnovers": "tov",
    "fouls": "pf",
}


def download_parquet(url, label):
    print(f"  {label}: {url.rsplit('/', 1)[-1]}", end="", flush=True)
    r = requests.get(url, timeout=120)
    if r.status_code == 404:
        raise SystemExit(f"\n  Not found: {url}\n  Check --season (ESPN uses the ENDING year).")
    r.raise_for_status()
    df = pd.read_parquet(io.BytesIO(r.content))
    print(f"  -> {len(df):,} rows, {len(r.content)/1024:.0f} KB")
    return df


def pct(made, att):
    """Shooting percentage, guarding against 0 attempts."""
    return (made / att.where(att != 0)).round(3).fillna(0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", default="2026",
                    help="ESPN season = ENDING year. 2025-26 season is 2026 (default: %(default)s)")
    ap.add_argument("--outdir", default="./nba_csv")
    ap.add_argument("--keep-allstar", action="store_true",
                    help="keep All-Star exhibition games (excluded by default)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"\nDownloading ESPN box scores for season {args.season} from GitHub\n" + "-" * 60)
    pb = download_parquet(PLAYER_URL.format(season=args.season), "player box")
    tb = download_parquet(TEAM_URL.format(season=args.season), "team box  ")

    # ---- Drop All-Star exhibitions -----------------------------------------
    # ESPN tags All-Star games as season_type=2, i.e. regular season. The rosters
    # ('Team Stars', 'Team Stripes', 'World') then pollute the teams dimension and
    # every per-team aggregate. Real franchises play 65+ games; All-Star squads
    # play at most 3, so a game-count threshold separates them robustly without
    # hardcoding names that change every year.
    if not args.keep_allstar:
        counts = tb.groupby("team_id")["game_id"].nunique()
        fake = set(counts[counts < 20].index)
        if fake:
            names = sorted(tb.loc[tb["team_id"].isin(fake), "team_display_name"].unique())
            drop_games = set(tb.loc[tb["team_id"].isin(fake), "game_id"])
            tb = tb[~tb["game_id"].isin(drop_games)]
            pb = pb[~pb["game_id"].isin(drop_games)]
            print(f"\n  Excluded {len(drop_games)} All-Star game(s): {', '.join(names)}")
            print("  (pass --keep-allstar to retain them)")

    print("\nBuilding tables\n" + "-" * 60)

    # ---- games: one row per game, from the team box (2 rows per game) --------
    home = tb[tb["team_home_away"] == "home"]
    away = tb[tb["team_home_away"] == "away"]
    games = (home[["game_id", "game_date", "season", "season_type", "team_id", "team_score"]]
             .rename(columns={"team_id": "home_team_id", "team_score": "home_pts"})
             .merge(away[["game_id", "team_id", "team_score"]]
                    .rename(columns={"team_id": "away_team_id", "team_score": "away_pts"}),
                    on="game_id", how="inner"))
    games["season_type"] = games["season_type"].map(SEASON_TYPE).fillna("Unknown")
    games["winner_team_id"] = games["home_team_id"].where(
        games["home_pts"] > games["away_pts"], games["away_team_id"])
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    # ---- teams: dimension, deduped from the team box ------------------------
    teams = (tb[["team_id", "team_display_name", "team_location", "team_name",
                 "team_abbreviation", "team_color"]]
             .drop_duplicates("team_id")
             .rename(columns={"team_display_name": "team_full_name",
                              "team_location": "team_city",
                              "team_name": "team_nickname"})
             .sort_values("team_id").reset_index(drop=True))

    # ---- team_game_stats: PK (game_id, team_id) -----------------------------
    tgs = tb[["game_id", "team_id", "team_winner", "team_score"] + list(TEAM_STATS)].copy()
    tgs = tgs.rename(columns={**TEAM_STATS, "team_score": "pts"})
    tgs["win_loss"] = tgs.pop("team_winner").map({True: "W", False: "L"})
    tgs["fg_pct"] = pct(tgs["fgm"], tgs["fga"])
    tgs["fg3_pct"] = pct(tgs["fg3m"], tgs["fg3a"])
    tgs["ft_pct"] = pct(tgs["ftm"], tgs["fta"])
    tgs = tgs.sort_values(["game_id", "team_id"]).reset_index(drop=True)

    # ---- players: PK player_id. Position is stable, so it lives HERE, not in
    #      player_game_stats. Note there is deliberately no team_id column.
    players = (pb[["athlete_id", "athlete_display_name", "athlete_position_abbreviation"]]
               .dropna(subset=["athlete_id"])
               .drop_duplicates("athlete_id")
               .rename(columns={"athlete_id": "player_id",
                                "athlete_display_name": "player_name",
                                "athlete_position_abbreviation": "position"})
               .sort_values("player_name").reset_index(drop=True))

    # ---- player_game_stats: PK (game_id, player_id). DNPs carry no stat line.
    active = pb[~pb["did_not_play"].fillna(False)].copy()
    pgs = active[["game_id", "athlete_id", "team_id", "starter"] + list(PLAYER_STATS)].copy()
    pgs = pgs.rename(columns={**PLAYER_STATS, "athlete_id": "player_id"})
    pgs["is_starter"] = pgs.pop("starter").astype("boolean").fillna(False).astype(int)
    pgs["fg_pct"] = pct(pgs["fgm"], pgs["fga"])
    pgs["fg3_pct"] = pct(pgs["fg3m"], pgs["fg3a"])
    pgs["ft_pct"] = pct(pgs["ftm"], pgs["fta"])
    pgs = pgs.dropna(subset=["player_id"]).sort_values(["game_id", "player_id"]).reset_index(drop=True)

    # ---- player_team_stints: why 'team' cannot be an attribute of a player ---
    stints = (active.merge(games[["game_id", "game_date"]], on="game_id", how="left",
                           suffixes=("", "_g"))
              .groupby(["athlete_id", "team_id"], as_index=False)
              .agg(games_played=("game_id", "nunique"),
                   first_game_date=("game_date", "min"),
                   last_game_date=("game_date", "max"))
              .rename(columns={"athlete_id": "player_id"})
              .sort_values(["player_id", "first_game_date"]).reset_index(drop=True))

    # ---- raw_player_game_wide: the unnormalized table students start from ----
    raw_wide = pb[[
        "game_id", "game_date", "season", "season_type",
        "athlete_id", "athlete_display_name", "athlete_position_abbreviation",
        "team_id", "team_display_name", "team_location", "team_name",
        "team_abbreviation", "team_color", "home_away", "team_score",
        "opponent_team_id", "opponent_team_display_name",
        "opponent_team_abbreviation", "opponent_team_score",
        "starter", "did_not_play",
    ] + list(PLAYER_STATS)].copy()
    raw_wide["season_type"] = raw_wide["season_type"].map(SEASON_TYPE).fillna("Unknown")

    outputs = {
        "raw_player_game_wide.csv": raw_wide,
        "teams.csv": teams,
        "players.csv": players,
        "games.csv": games,
        "team_game_stats.csv": tgs,
        "player_game_stats.csv": pgs,
        "player_team_stints.csv": stints,
    }
    for name, df in outputs.items():
        df.to_csv(os.path.join(args.outdir, name), index=False)
        print(f"  {name:<28} {len(df):>7,} rows  x {len(df.columns):>2} cols")

    print("\nIntegrity checks\n" + "-" * 60)
    for label, ok in [
        ("(game_id, team_id) is unique", not tgs.duplicated(["game_id", "team_id"]).any()),
        ("(game_id, player_id) is unique", not pgs.duplicated(["game_id", "player_id"]).any()),
        ("every game has exactly 2 team rows", (tgs.groupby("game_id").size() == 2).all()),
        ("player FKs resolve", pgs["player_id"].isin(players["player_id"]).all()),
        ("game FKs resolve (players)", pgs["game_id"].isin(games["game_id"]).all()),
        ("game FKs resolve (teams)", tgs["game_id"].isin(games["game_id"]).all()),
        ("team FKs resolve", tgs["team_id"].isin(teams["team_id"]).all()),
        ("exactly 30 teams", len(teams) == 30),
        ("every team played 65+ games", (tgs.groupby("team_id").size() >= 65).all()),
    ]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    multi = stints["player_id"].duplicated().sum()
    print(f"\n  Games: {len(games):,}  ({games['game_date'].min()} to {games['game_date'].max()})")
    print("  By type: " + ", ".join(
        f"{k}={v}" for k, v in games["season_type"].value_counts().items()))
    print(f"  {multi} players appeared for more than one team -- your 3NF example.")
    print(f"\nDone. CSVs in {os.path.abspath(args.outdir)}\n")


if __name__ == "__main__":
    sys.exit(main())
