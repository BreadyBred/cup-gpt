# Cup-GPT

World Cup 2026 prediction system. Trains an XGBoost model on 150+ years of international football results, then runs 100,000 Monte Carlo simulations of the full tournament bracket to estimate each team's probability of winning, reaching the final, semifinals, and beyond.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/predict.py
```

The pipeline runs end-to-end:

1. **Data** — downloads historical international match results (~47k matches)
2. **Features** — computes Elo ratings, head-to-head records, recent form, goal averages, and confederation data for every match
3. **Training** — fits an XGBoost classifier (3-class: home win / draw / away win) on the full dataset
4. **Simulation** — simulates the 2026 World Cup bracket 100,000 times using the trained model's match probabilities
5. **Report** — generates a self-contained HTML report

## Output

`output/report.html` — dark-themed, self-contained report with:

- Written analysis of favorites, dark horses, and dangerous groups
- Horizontal bar charts (win % and semifinal %) for the top 16 teams
- Sortable table of all 48 teams with per-phase probabilities
