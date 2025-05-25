"""Feature engineering pipeline for World Cup 2026 prediction.

Computes Elo ratings, head-to-head records, recent form, and other
match-level features from historical international football results.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── confederation mapping ────────────────────────────────────────────────────

CONFEDERATIONS = {
    # UEFA
    "Albania": "UEFA", "Andorra": "UEFA", "Armenia": "UEFA", "Austria": "UEFA",
    "Azerbaijan": "UEFA", "Belarus": "UEFA", "Belgium": "UEFA",
    "Bosnia and Herzegovina": "UEFA", "Bulgaria": "UEFA", "Croatia": "UEFA",
    "Cyprus": "UEFA", "Czech Republic": "UEFA", "Czechia": "UEFA",
    "Denmark": "UEFA", "England": "UEFA", "Estonia": "UEFA",
    "Faroe Islands": "UEFA", "Finland": "UEFA", "France": "UEFA",
    "Georgia": "UEFA", "Germany": "UEFA", "Gibraltar": "UEFA",
    "Greece": "UEFA", "Hungary": "UEFA", "Iceland": "UEFA",
    "Republic of Ireland": "UEFA", "Ireland": "UEFA",
    "Israel": "UEFA", "Italy": "UEFA", "Kazakhstan": "UEFA", "Kosovo": "UEFA",
    "Latvia": "UEFA", "Liechtenstein": "UEFA", "Lithuania": "UEFA",
    "Luxembourg": "UEFA", "Malta": "UEFA", "Moldova": "UEFA",
    "Montenegro": "UEFA", "Netherlands": "UEFA", "North Macedonia": "UEFA",
    "Northern Ireland": "UEFA", "Norway": "UEFA", "Poland": "UEFA",
    "Portugal": "UEFA", "Romania": "UEFA", "Russia": "UEFA",
    "San Marino": "UEFA", "Scotland": "UEFA", "Serbia": "UEFA",
    "Serbia and Montenegro": "UEFA", "Slovakia": "UEFA", "Slovenia": "UEFA",
    "Spain": "UEFA", "Sweden": "UEFA", "Switzerland": "UEFA",
    "Turkey": "UEFA", "Ukraine": "UEFA", "Wales": "UEFA",
    "Soviet Union": "UEFA", "Czechoslovakia": "UEFA", "Yugoslavia": "UEFA",
    "East Germany": "UEFA", "West Germany": "UEFA",
    # CONMEBOL
    "Argentina": "CONMEBOL", "Bolivia": "CONMEBOL", "Brazil": "CONMEBOL",
    "Chile": "CONMEBOL", "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL", "Peru": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    # CONCACAF
    "Antigua and Barbuda": "CONCACAF", "Aruba": "CONCACAF",
    "Bahamas": "CONCACAF", "Barbados": "CONCACAF", "Belize": "CONCACAF",
    "Bermuda": "CONCACAF", "Canada": "CONCACAF", "Cayman Islands": "CONCACAF",
    "Costa Rica": "CONCACAF", "Cuba": "CONCACAF",
    "Curaçao": "CONCACAF", "Curacao": "CONCACAF",
    "Dominica": "CONCACAF", "Dominican Republic": "CONCACAF",
    "El Salvador": "CONCACAF", "Grenada": "CONCACAF",
    "Guatemala": "CONCACAF", "Guyana": "CONCACAF", "Haiti": "CONCACAF",
    "Honduras": "CONCACAF", "Jamaica": "CONCACAF", "Mexico": "CONCACAF",
    "Montserrat": "CONCACAF", "Nicaragua": "CONCACAF", "Panama": "CONCACAF",
    "Puerto Rico": "CONCACAF",
    "Saint Kitts and Nevis": "CONCACAF", "Saint Lucia": "CONCACAF",
    "Saint Vincent and the Grenadines": "CONCACAF",
    "Suriname": "CONCACAF", "Trinidad and Tobago": "CONCACAF",
    "Turks and Caicos Islands": "CONCACAF",
    "United States": "CONCACAF", "US Virgin Islands": "CONCACAF",
    # AFC
    "Afghanistan": "AFC", "Australia": "AFC", "Bahrain": "AFC",
    "Bangladesh": "AFC", "Bhutan": "AFC", "Brunei": "AFC",
    "Cambodia": "AFC", "China PR": "AFC", "Chinese Taipei": "AFC",
    "Guam": "AFC", "Hong Kong": "AFC", "India": "AFC",
    "Indonesia": "AFC", "Iran": "AFC", "IR Iran": "AFC",
    "Iraq": "AFC", "Japan": "AFC", "Jordan": "AFC",
    "Kuwait": "AFC", "Kyrgyzstan": "AFC", "Laos": "AFC",
    "Lebanon": "AFC", "Macau": "AFC", "Malaysia": "AFC",
    "Maldives": "AFC", "Mongolia": "AFC", "Myanmar": "AFC",
    "Nepal": "AFC", "North Korea": "AFC", "Oman": "AFC",
    "Pakistan": "AFC", "Palestine": "AFC", "Philippines": "AFC",
    "Qatar": "AFC", "Saudi Arabia": "AFC", "Singapore": "AFC",
    "South Korea": "AFC", "Korea Republic": "AFC",
    "Sri Lanka": "AFC", "Syria": "AFC", "Tajikistan": "AFC",
    "Thailand": "AFC", "Timor-Leste": "AFC", "Turkmenistan": "AFC",
    "United Arab Emirates": "AFC", "Uzbekistan": "AFC",
    "Vietnam": "AFC", "Yemen": "AFC",
    # CAF
    "Algeria": "CAF", "Angola": "CAF", "Benin": "CAF", "Botswana": "CAF",
    "Burkina Faso": "CAF", "Burundi": "CAF", "Cameroon": "CAF",
    "Cape Verde": "CAF", "Cabo Verde": "CAF",
    "Central African Republic": "CAF", "Chad": "CAF",
    "Comoros": "CAF", "Congo": "CAF", "Congo DR": "CAF", "DR Congo": "CAF",
    "Côte d'Ivoire": "CAF", "Ivory Coast": "CAF",
    "Djibouti": "CAF", "Egypt": "CAF", "Equatorial Guinea": "CAF",
    "Eritrea": "CAF", "Eswatini": "CAF", "Swaziland": "CAF",
    "Ethiopia": "CAF", "Gabon": "CAF", "Gambia": "CAF", "Ghana": "CAF",
    "Guinea": "CAF", "Guinea-Bissau": "CAF", "Kenya": "CAF",
    "Lesotho": "CAF", "Liberia": "CAF", "Libya": "CAF",
    "Madagascar": "CAF", "Malawi": "CAF", "Mali": "CAF",
    "Mauritania": "CAF", "Mauritius": "CAF", "Morocco": "CAF",
    "Mozambique": "CAF", "Namibia": "CAF", "Niger": "CAF",
    "Nigeria": "CAF", "Rwanda": "CAF",
    "São Tomé and Príncipe": "CAF", "Sao Tome and Principe": "CAF",
    "Senegal": "CAF", "Seychelles": "CAF", "Sierra Leone": "CAF",
    "Somalia": "CAF", "South Africa": "CAF", "South Sudan": "CAF",
    "Sudan": "CAF", "Tanzania": "CAF", "Togo": "CAF", "Tunisia": "CAF",
    "Uganda": "CAF", "Zambia": "CAF", "Zimbabwe": "CAF",
    # OFC
    "American Samoa": "OFC", "Cook Islands": "OFC", "Fiji": "OFC",
    "New Caledonia": "OFC", "New Zealand": "OFC",
    "Papua New Guinea": "OFC", "Samoa": "OFC", "Solomon Islands": "OFC",
    "Tahiti": "OFC", "Tonga": "OFC", "Vanuatu": "OFC",
}

CONF_ENCODING = {
    "UEFA": 0, "CONMEBOL": 1, "CONCACAF": 2,
    "AFC": 3, "CAF": 4, "OFC": 5, "UNKNOWN": 6,
}

STAGE_ENCODING = {
    "group": 1, "R32": 2, "R16": 3, "QF": 4, "SF": 5, "3rd": 5, "Final": 6,
}

FEATURE_COLUMNS = [
    "home_elo", "away_elo", "elo_diff",
    "h2h_win_rate",
    "home_form", "away_form",
    "home_goals_scored", "home_goals_conceded",
    "away_goals_scored", "away_goals_conceded",
    "is_world_cup", "is_qualifier", "is_friendly",
    "is_neutral",
    "home_confederation", "away_confederation",
    "stage",
]


class FeatureEngine:
    """Maintains running match statistics and builds feature vectors."""

    def __init__(self):
        self.elo: dict[str, float] = defaultdict(lambda: 1500.0)
        self.matches: dict[str, list[dict]] = defaultdict(list)
        self.h2h: dict[tuple[str, str], list[dict]] = defaultdict(list)

    # ── lookups ───────────────────────────────────────────────────────────

    @staticmethod
    def _conf_code(team: str) -> int:
        return CONF_ENCODING.get(CONFEDERATIONS.get(team, "UNKNOWN"), 6)

    def _form(self, team: str, n: int = 10) -> float:
        recent = self.matches[team][-n:]
        if not recent:
            return 0.5
        return sum(m["pts"] for m in recent) / len(recent)

    def _avg_goals_scored(self, team: str, n: int = 10) -> float:
        recent = self.matches[team][-n:]
        if not recent:
            return 0.0
        return sum(m["gf"] for m in recent) / len(recent)

    def _avg_goals_conceded(self, team: str, n: int = 10) -> float:
        recent = self.matches[team][-n:]
        if not recent:
            return 0.0
        return sum(m["ga"] for m in recent) / len(recent)

    def _h2h_rate(self, team_a: str, team_b: str, n: int = 10) -> float:
        key = tuple(sorted([team_a, team_b]))
        records = self.h2h[key][-n:]
        if not records:
            return 0.5
        wins = sum(1 for r in records if r["winner"] == team_a)
        draws = sum(1 for r in records if r["winner"] is None)
        return (wins + 0.5 * draws) / len(records)

    # ── feature vector ────────────────────────────────────────────────────

    def _compute_features(
        self,
        home: str,
        away: str,
        tournament: str,
        neutral: bool,
        stage: int = 0,
    ) -> dict:
        h_elo = self.elo[home]
        a_elo = self.elo[away]
        is_wc = int(tournament == "FIFA World Cup")
        is_qual = int("qualification" in tournament.lower())
        is_fri = int("friendly" in tournament.lower())

        return {
            "home_elo": h_elo,
            "away_elo": a_elo,
            "elo_diff": h_elo - a_elo,
            "h2h_win_rate": self._h2h_rate(home, away),
            "home_form": self._form(home),
            "away_form": self._form(away),
            "home_goals_scored": self._avg_goals_scored(home),
            "home_goals_conceded": self._avg_goals_conceded(home),
            "away_goals_scored": self._avg_goals_scored(away),
            "away_goals_conceded": self._avg_goals_conceded(away),
            "is_world_cup": is_wc,
            "is_qualifier": is_qual,
            "is_friendly": is_fri,
            "is_neutral": int(neutral),
            "home_confederation": self._conf_code(home),
            "away_confederation": self._conf_code(away),
            "stage": stage,
        }

    # ── state update ──────────────────────────────────────────────────────

    def _update(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        tournament: str,
    ) -> None:
        if home_score > away_score:
            h_pts, a_pts, winner = 1.0, 0.0, home
        elif home_score < away_score:
            h_pts, a_pts, winner = 0.0, 1.0, away
        else:
            h_pts, a_pts, winner = 0.5, 0.5, None

        k = 30 if tournament == "FIFA World Cup" else 20
        exp_h = 1.0 / (1.0 + 10.0 ** ((self.elo[away] - self.elo[home]) / 400.0))
        exp_a = 1.0 - exp_h
        self.elo[home] += k * (h_pts - exp_h)
        self.elo[away] += k * (a_pts - exp_a)

        self.matches[home].append({"gf": home_score, "ga": away_score, "pts": h_pts})
        self.matches[away].append({"gf": away_score, "ga": home_score, "pts": a_pts})

        key = tuple(sorted([home, away]))
        self.h2h[key].append({"winner": winner})

    # ── public API ────────────────────────────────────────────────────────

    def build_training_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Walk through all matches chronologically and produce (X, y)."""
        df = df.sort_values("date").reset_index(drop=True)

        features: list[dict] = []
        labels: list[int] = []

        for _, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            h_score = row["home_score"]
            a_score = row["away_score"]
            tournament = row["tournament"]
            neutral = row.get("neutral", False)

            if pd.isna(h_score) or pd.isna(a_score):
                continue

            h_score = int(h_score)
            a_score = int(a_score)

            if isinstance(neutral, str):
                neutral = neutral.strip().upper() == "TRUE"

            feat = self._compute_features(home, away, tournament, neutral)
            features.append(feat)

            if h_score > a_score:
                labels.append(0)
            elif h_score == a_score:
                labels.append(1)
            else:
                labels.append(2)

            self._update(home, away, h_score, a_score, tournament)

        X = pd.DataFrame(features, columns=FEATURE_COLUMNS)
        y = np.array(labels)
        print(f"Built {len(X)} feature vectors from {len(df)} matches")
        return X, y

    def match_features_dict(
        self,
        home: str,
        away: str,
        stage: str = "group",
        neutral: bool = True,
    ) -> dict:
        """Build a feature dict for a single upcoming match (no DataFrame overhead)."""
        stage_code = STAGE_ENCODING.get(stage, 0)
        return self._compute_features(home, away, "FIFA World Cup", neutral, stage_code)

    def predict_features(
        self,
        home: str,
        away: str,
        stage: str = "group",
        neutral: bool = True,
    ) -> pd.DataFrame:
        """Build a feature row for a single upcoming match (no state update)."""
        stage_code = STAGE_ENCODING.get(stage, 0)
        feat = self._compute_features(home, away, "FIFA World Cup", neutral, stage_code)
        return pd.DataFrame([feat], columns=FEATURE_COLUMNS)


def load_and_preprocess() -> tuple[pd.DataFrame, np.ndarray, "FeatureEngine"]:
    """Convenience: load CSV, build features, return (X, y, engine)."""
    csv_path = DATA_DIR / "results.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found — run fetch_data.py first")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError as e:
        print(f"Error: failed to parse {csv_path} — {e}")
        sys.exit(1)

    print(f"Loaded {len(df)} historical matches")
    engine = FeatureEngine()
    X, y = engine.build_training_data(df)
    return X, y, engine


if __name__ == "__main__":
    X, y, engine = load_and_preprocess()
    print(f"Features shape: {X.shape}")
    print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    top = sorted(engine.elo.items(), key=lambda x: -x[1])[:20]
    print("Top 20 Elo ratings:")
    for team, rating in top:
        print(f"  {team:25s} {rating:.0f}")
