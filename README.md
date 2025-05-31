# Cup-GPT

World Cup 2026 prediction system. Trains an XGBoost model on ~50k international football matches going back to 1872, then runs 100,000 Monte Carlo simulations of the full 48-team tournament bracket.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/predict.py
```

This runs the full pipeline:

1. Downloads historical international match results if not already cached
2. Computes goal-margin adjusted Elo ratings, head-to-head records, recent form, goal averages, Elo momentum and confederation data
3. Trains an XGBoost classifier (home win / draw / away win) on post-2000 matches (Elo warms up on the full history)
4. Simulates the 2026 World Cup bracket 100,000 times using model probabilities
5. Generates a self-contained HTML report

## Results

The simulation is fully deterministic (fixed random seed), so running it twice on the same data and model produces identical output.

Top 5 predicted winners:

| Team | Win % | Final % | SF % |
|------|-------|---------|------|
| Spain | 16.8 | 25.6 | 38.1 |
| Argentina | 14.4 | 23.3 | 37.4 |
| France | 13.4 | 23.4 | 35.1 |
| Germany | 6.7 | 13.8 | 24.8 |
| Brazil | 6.3 | 12.7 | 23.0 |

Model accuracy: 59.2% (3-class), log-loss: 0.8746 on a held-out 20% test set.

## Output

`output/report.html` is a self-contained dark-themed report with:

- Written analysis of favorites, dark horses and dangerous groups
- Predicted group standings with advance probabilities
- Full predicted bracket from R32 to the Final
- Bar charts for win % and semifinal % (top 16 teams)
- Sortable table of all 48 teams with per-phase probabilities

## How it works

**Elo ratings** are computed dynamically from all historical matches. The K-factor is 30 for World Cup matches, 20 otherwise, scaled by goal margin (`K * (1 + 0.5 * ln(margin + 1))`). This means a 4-0 win moves the ratings more than a 1-0 win.

**Features per match** (18 total): home/away Elo, Elo difference, Elo momentum (trend over last 10 games), head-to-head win rate (last 10 meetings), recent form, average goals scored/conceded, tournament type flags, neutral venue, and confederation encoding.

**Training** uses only matches from 2000 onwards to keep the distribution modern, but Elo ratings warm up on the full 150-year history so they're accurate by the time modern matches start.

**Simulation** precomputes win probabilities for all 1,128 possible team pairings, then samples 100k tournament runs. Group matches can end in draws; knockout matches split draw probability 50/50 for advancement. Third-placed teams are assigned to R32 slots using backtracking constraint satisfaction matching FIFA's bracket rules.
