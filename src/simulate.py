"""Monte Carlo simulation of the 2026 FIFA World Cup bracket."""

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import FEATURE_COLUMNS, FeatureEngine, load_and_preprocess

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "xgb_model.json"
OUTPUT_DIR = ROOT / "output"
N_ITER = 100_000

# ── 2026 FIFA World Cup groups (official draw, 5 Dec 2025) ───────────────────

GROUPS = {
    "A": ["Mexico", "South Korea", "South Africa", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Ecuador", "Côte d'Ivoire", "Curaçao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Panama", "Ghana"],
}

# ── R32 bracket (FIFA confirmed) ────────────────────────────────────────────
# Fixed pairings: position code "1X" = group X winner, "2X" = runner-up

R32_FIXED = {
    73: ("2A", "2B"), 75: ("1F", "2C"), 76: ("1C", "2F"), 78: ("2E", "2I"),
    83: ("2K", "2L"), 84: ("1H", "2J"), 86: ("1J", "2H"), 88: ("2D", "2G"),
}

# Slots for 3rd-placed teams: match_id → (group-winner position, eligible source groups)
R32_THIRD = {
    74: ("1E", "ABCDF"), 77: ("1I", "CDFGH"), 79: ("1A", "CEFHI"),
    80: ("1L", "EHIJK"), 81: ("1D", "BEFIJ"), 82: ("1G", "AEHIJ"),
    85: ("1B", "EFGIJ"), 87: ("1K", "DEIJL"),
}

# How R32 winners feed into R16 (pairs of R32 match IDs)
R16_FLOW = [(74, 77), (73, 75), (76, 78), (79, 80),
            (83, 84), (81, 82), (86, 88), (85, 87)]

QF_FLOW = [(0, 1), (2, 3), (4, 5), (6, 7)]
SF_FLOW = [(0, 1), (2, 3)]


# ── probability cache ────────────────────────────────────────────────────────

def precompute(engine: FeatureEngine, model) -> dict:
    all_teams = sorted({t for g in GROUPS.values() for t in g})
    pairs = list(combinations(all_teams, 2))

    rows, keys = [], []
    for ta, tb in pairs:
        rows.append(engine.match_features_dict(ta, tb, True))
        keys.append((ta, tb))

    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    probas = model.predict_proba(X)
    return dict(zip(keys, probas))


def mp(cache, ta, tb):
    """Look up (P_ta_win, P_draw, P_tb_win), handling team order."""
    if ta <= tb:
        return cache[(ta, tb)]
    p = cache[(tb, ta)]
    return np.array([p[2], p[1], p[0]])


# ── simulation helpers ───────────────────────────────────────────────────────

def _score(rng, outcome):
    """Generate a plausible scoreline for a match outcome (0=home, 1=draw, 2=away)."""
    if outcome == 0:
        h = max(1, int(rng.poisson(1.4)))
        a = min(int(rng.poisson(0.6)), h - 1)
    elif outcome == 1:
        h = a = int(rng.poisson(1.0))
    else:
        a = max(1, int(rng.poisson(1.4)))
        h = min(int(rng.poisson(0.6)), a - 1)
    return h, a


def _sim_groups(cache, rng):
    standings = {}
    for g, teams in GROUPS.items():
        pts, gd, gs = defaultdict(int), defaultdict(int), defaultdict(int)
        for ta, tb in combinations(teams, 2):
            p = mp(cache, ta, tb)
            r = rng.random()
            if r < p[0]:
                out = 0
                pts[ta] += 3
            elif r < p[0] + p[1]:
                out = 1
                pts[ta] += 1
                pts[tb] += 1
            else:
                out = 2
                pts[tb] += 3
            h, a = _score(rng, out)
            gd[ta] += h - a
            gd[tb] += a - h
            gs[ta] += h
            gs[tb] += a

        ranked = sorted(
            teams,
            key=lambda t: (pts[t], gd[t], gs[t], rng.random()),
            reverse=True,
        )
        standings[g] = [
            {"team": t, "pts": pts[t], "gd": gd[t], "gs": gs[t]} for t in ranked
        ]
    return standings


def _best_thirds(standings):
    thirds = []
    for g, ranks in standings.items():
        entry = dict(ranks[2])
        entry["group"] = g
        thirds.append(entry)
    thirds.sort(key=lambda t: (t["pts"], t["gd"], t["gs"]), reverse=True)
    return thirds[:8]


def _assign_thirds(qualifying):
    """Map 8 best third-placed teams to R32 slots via backtracking search."""
    slot_data = [
        (74, set("ABCDF")), (77, set("CDFGH")), (79, set("CEFHI")),
        (80, set("EHIJK")), (81, set("BEFIJ")), (82, set("AEHIJ")),
        (85, set("EFGIJ")), (87, set("DEIJL")),
    ]
    by_group = {t["group"]: t["team"] for t in qualifying}
    q_groups = set(by_group.keys())

    slots = sorted(slot_data, key=lambda s: len(s[1] & q_groups))

    def solve(idx, remaining):
        if idx == len(slots):
            return {}
        mid, eligible = slots[idx]
        for group in sorted(eligible & remaining):
            result = solve(idx + 1, remaining - {group})
            if result is not None:
                result[mid] = by_group[group]
                return result
        return None

    return solve(0, q_groups) or {}


def _ko(cache, rng, ta, tb):
    """Simulate a knockout match — draws resolved as 50/50 advancement."""
    p = mp(cache, ta, tb)
    p_a = p[0] + p[1] * 0.5
    return ta if rng.random() < p_a else tb


# ── main simulation loop ────────────────────────────────────────────────────

def simulate(engine, model, n=N_ITER):
    all_teams = [t for g in GROUPS.values() for t in g]

    print("Precomputing match probabilities...")
    cache = precompute(engine, model)
    print(f"  {len(cache):,} entries cached")

    cnt = {t: defaultdict(int) for t in all_teams}
    gp = {g: {t: [0, 0, 0, 0] for t in teams} for g, teams in GROUPS.items()}
    rng = np.random.default_rng(42)

    print(f"Running {n:,} iterations...")
    for i in range(n):
        if (i + 1) % 20_000 == 0:
            print(f"  {i + 1:>7,} / {n:,}")

        # ── group stage ──
        st = _sim_groups(cache, rng)
        for g, ranks in st.items():
            for pos, entry in enumerate(ranks):
                gp[g][entry["team"]][pos] += 1
        w = {g: r[0]["team"] for g, r in st.items()}
        ru = {g: r[1]["team"] for g, r in st.items()}

        bt = _best_thirds(st)
        qualified = set()
        for g in GROUPS:
            qualified.add(w[g])
            qualified.add(ru[g])
        for t in bt:
            qualified.add(t["team"])

        for t in all_teams:
            if t not in qualified:
                cnt[t]["ge"] += 1

        # ── R32 ──
        tm = _assign_thirds(bt)

        if len(tm) != 8:
            print("Third-place assignment failed!")
            print("Best thirds:", [t["group"] for t in bt])
            print("Assignment:", tm)
            raise RuntimeError("Could not assign all third-place teams")

        def pos(code):
            return w[code[1]] if code[0] == "1" else ru[code[1]]

        r32w = {}
        for mid, (pa, pb) in R32_FIXED.items():
            r32w[mid] = _ko(cache, rng, pos(pa), pos(pb))
        for mid, (wp, _) in R32_THIRD.items():
            r32w[mid] = _ko(cache, rng, pos(wp), tm[mid])

        r32_adv = set(r32w.values())
        for t in qualified:
            if t not in r32_adv:
                cnt[t]["r32"] += 1

        # ── R16 ──
        r16w = {}
        for idx, (ma, mb) in enumerate(R16_FLOW):
            r16w[idx] = _ko(cache, rng, r32w[ma], r32w[mb])

        # ── QF ──
        qfw = {}
        for idx, (a, b) in enumerate(QF_FLOW):
            winner = _ko(cache, rng, r16w[a], r16w[b])
            loser = r16w[b] if winner == r16w[a] else r16w[a]
            qfw[idx] = winner
            cnt[loser]["qf"] += 1

        # ── SF ──
        sfw = {}
        sf_losers = []
        for idx, (a, b) in enumerate(SF_FLOW):
            winner = _ko(cache, rng, qfw[a], qfw[b])
            loser = qfw[b] if winner == qfw[a] else qfw[a]
            sfw[idx] = winner
            sf_losers.append(loser)
            cnt[loser]["sf"] += 1

        # ── Final ──
        champ = _ko(cache, rng, sfw[0], sfw[1])
        runner = sfw[1] if champ == sfw[0] else sfw[0]
        cnt[champ]["win"] += 1
        cnt[runner]["fin"] += 1

    # ── aggregate results ──
    results = []
    for t in all_teams:
        c = cnt[t]
        reach_final = c["fin"] + c["win"]
        reach_sf = c["sf"] + reach_final
        reach_qf = c["qf"] + reach_sf
        results.append({
            "team": t,
            "win_pct": round(c["win"] / n * 100, 2),
            "final_pct": round(reach_final / n * 100, 2),
            "sf_pct": round(reach_sf / n * 100, 2),
            "qf_pct": round(reach_qf / n * 100, 2),
            "r32_exit_pct": round(c["r32"] / n * 100, 2),
            "group_exit_pct": round(c["ge"] / n * 100, 2),
        })
    results.sort(key=lambda x: -x["win_pct"])

    print("\nTop 10:")
    for r in results[:10]:
        print(
            f"  {r['team']:25s}  Win {r['win_pct']:5.1f}%"
            f"   Final {r['final_pct']:5.1f}%   SF {r['sf_pct']:5.1f}%"
        )

    # ── group standings from MC ──
    group_standings = {}
    for g in GROUPS:
        st_list = []
        for t in GROUPS[g]:
            st_list.append({
                "team": t,
                "1st_pct": round(gp[g][t][0] / n * 100, 1),
                "2nd_pct": round(gp[g][t][1] / n * 100, 1),
                "3rd_pct": round(gp[g][t][2] / n * 100, 1),
                "4th_pct": round(gp[g][t][3] / n * 100, 1),
            })
        st_list.sort(key=lambda x: x["1st_pct"], reverse=True)
        group_standings[g] = st_list

    # ── most-likely bracket ──
    bracket = _build_predicted_bracket(gp, cache, results, n)

    return results, group_standings, bracket


# ── predicted bracket builder ────────────────────────────────────────────────

def _build_predicted_bracket(gp, cache, results, n):
    """Build the single most-likely bracket path from MC group data + model."""
    by_team = {r["team"]: r for r in results}

    # Most likely group order: 1st by max-1st-count, 2nd by max-2nd among rest, etc.
    pred_w, pred_ru, pred_3rd = {}, {}, {}
    third_info = []

    for g in GROUPS:
        pool = list(GROUPS[g])
        first = max(pool, key=lambda t: gp[g][t][0])
        pool.remove(first)
        second = max(pool, key=lambda t: gp[g][t][1])
        pool.remove(second)
        third = max(pool, key=lambda t: gp[g][t][2])
        pred_w[g] = first
        pred_ru[g] = second
        pred_3rd[g] = third
        third_info.append((g, third, by_team[third]["group_exit_pct"]))

    # Best 8 thirds: lowest group-exit rate
    third_info.sort(key=lambda x: x[2])
    qual_thirds = [{"group": g, "team": t, "pts": 0, "gd": 0, "gs": 0}
                   for g, t, _ in third_info[:8]]
    third_map = _assign_thirds(qual_thirds)

    def pos(code):
        return pred_w[code[1]] if code[0] == "1" else pred_ru[code[1]]

    def match(ta, tb):
        p = mp(cache, ta, tb)
        pa = float(p[0] + p[1] * 0.5)
        winner = ta if pa >= 0.5 else tb
        return {"a": ta, "b": tb, "a_pct": round(pa * 100, 1), "winner": winner}

    # R32
    r32 = {}
    for mid, (pa, pb) in R32_FIXED.items():
        r32[mid] = match(pos(pa), pos(pb))
    for mid, (wp, _) in R32_THIRD.items():
        r32[mid] = match(pos(wp), third_map[mid])

    # R16
    r16 = {}
    for idx, (ma, mb) in enumerate(R16_FLOW):
        r16[idx] = match(r32[ma]["winner"], r32[mb]["winner"])

    # QF
    qf = {}
    for idx, (a, b) in enumerate(QF_FLOW):
        qf[idx] = match(r16[a]["winner"], r16[b]["winner"])

    # SF
    sf = {}
    for idx, (a, b) in enumerate(SF_FLOW):
        sf[idx] = match(qf[a]["winner"], qf[b]["winner"])

    # Final
    final = match(sf[0]["winner"], sf[1]["winner"])

    # Serialise in match-number order
    r32_list = [r32[mid] for mid in sorted(r32)]
    r16_list = [r16[i] for i in range(len(r16))]
    qf_list = [qf[i] for i in range(len(qf))]
    sf_list = [sf[i] for i in range(len(sf))]

    return {"r32": r32_list, "r16": r16_list, "qf": qf_list, "sf": sf_list, "final": final}


# ── entry point ──────────────────────────────────────────────────────────────

def run():
    if not MODEL_PATH.exists():
        print(f"Error: model not found at {MODEL_PATH} — run train.py first")
        sys.exit(1)

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    _, _, engine = load_and_preprocess()
    results, group_standings, bracket = simulate(engine, model)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "simulation_results.json"
    payload = {
        "n_iterations": N_ITER,
        "groups": {g: list(teams) for g, teams in GROUPS.items()},
        "results": results,
        "group_standings": group_standings,
        "bracket": bracket,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {out}")
    return results


if __name__ == "__main__":
    run()
