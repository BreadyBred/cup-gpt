"""Self-contained HTML report generator for World Cup 2026 predictions."""

import json
import sys
from collections import defaultdict
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
RESULTS_PATH = OUTPUT_DIR / "simulation_results.json"

TEAM_CONF = {
    "Mexico": "CONCACAF", "South Korea": "AFC", "South Africa": "CAF",
    "Czech Republic": "UEFA", "Canada": "CONCACAF", "Switzerland": "UEFA",
    "Qatar": "AFC", "Bosnia and Herzegovina": "UEFA", "Brazil": "CONMEBOL",
    "Morocco": "CAF", "Scotland": "UEFA", "Haiti": "CONCACAF",
    "United States": "CONCACAF", "Paraguay": "CONMEBOL", "Australia": "AFC",
    "Turkey": "UEFA", "Germany": "UEFA", "Ecuador": "CONMEBOL",
    "Côte d'Ivoire": "CAF", "Curaçao": "CONCACAF", "Netherlands": "UEFA",
    "Japan": "AFC", "Tunisia": "CAF", "Sweden": "UEFA", "Belgium": "UEFA",
    "Iran": "AFC", "Egypt": "CAF", "New Zealand": "OFC", "Spain": "UEFA",
    "Uruguay": "CONMEBOL", "Saudi Arabia": "AFC", "Cape Verde": "CAF",
    "France": "UEFA", "Senegal": "CAF", "Norway": "UEFA", "Iraq": "AFC",
    "Argentina": "CONMEBOL", "Austria": "UEFA", "Algeria": "CAF",
    "Jordan": "AFC", "Portugal": "UEFA", "Colombia": "CONMEBOL",
    "Uzbekistan": "AFC", "DR Congo": "CAF", "England": "UEFA",
    "Croatia": "UEFA", "Panama": "CONCACAF", "Ghana": "CAF",
}


def load_results():
    if not RESULTS_PATH.exists():
        print(f"Error: {RESULTS_PATH} not found — run simulate.py first")
        sys.exit(1)
    try:
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Error reading {RESULTS_PATH}: {e}")
        sys.exit(1)


# ── analysis prose ───────────────────────────────────────────────────────────

def _build_analysis(results, groups):
    sr = sorted(results, key=lambda x: -x["win_pct"])
    top3 = sr[:3]

    dark_horses = [r for r in sr[3:] if r["win_pct"] >= 5.0]
    if len(dark_horses) < 2:
        dark_horses = sr[3:6]
    else:
        dark_horses = dark_horses[:3]

    paras = []

    paras.append(
        f"After running 100,000 Monte Carlo simulations of the 2026 FIFA World Cup, "
        f"<strong>{top3[0]['team']}</strong> emerges as the tournament favorite with a "
        f"<strong>{top3[0]['win_pct']:.1f}%</strong> probability of lifting the trophy. "
        f"{top3[1]['team']} is projected at {top3[1]['win_pct']:.1f}%, with {top3[2]['team']} "
        f"close behind at {top3[2]['win_pct']:.1f}%. {top3[0]['team']}'s model strength "
        f"is reflected across all tournament stages: a {top3[0]['final_pct']:.1f}% chance of "
        f"reaching the final and {top3[0]['sf_pct']:.1f}% of reaching the semifinals."
    )

    if dark_horses:
        names = [f"{d['team']} ({d['win_pct']:.1f}%)" for d in dark_horses]
        joined = ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]
        p2 = (
            f"Beyond the top tier, several dark horses could upset the established order. "
            f"{joined} all carry meaningful win probabilities. "
        )
        best_dh = dark_horses[0]
        if best_dh["sf_pct"] >= 15:
            p2 += (
                f"{best_dh['team']} stands out with a {best_dh['sf_pct']:.1f}% semifinal "
                f"probability, making them a serious knockout-stage threat."
            )
        paras.append(p2)

    danger = []
    for g, teams in groups.items():
        by_team = {r["team"]: r for r in results}
        sf_vals = [(t, by_team[t]["sf_pct"]) for t in teams if t in by_team]
        sf_vals.sort(key=lambda x: -x[1])
        if len(sf_vals) >= 2 and sf_vals[0][1] > 20 and sf_vals[1][1] > 12:
            danger.append((g, sf_vals[0], sf_vals[1]))
    danger.sort(key=lambda x: -(x[1][1] + x[2][1]))

    if danger:
        dg = danger[0]
        p3 = (
            f"Group {dg[0]} is the most treacherous draw, pairing {dg[1][0]} "
            f"({dg[1][1]:.1f}% SF probability) with {dg[2][0]} ({dg[2][1]:.1f}%). "
            f"One of these contenders risks an early exit or a tougher knockout path as "
            f"a third-placed qualifier. "
        )
        if len(danger) > 1:
            dg2 = danger[1]
            p3 += (
                f"Group {dg2[0]} presents a similar dilemma, with {dg2[1][0]} and "
                f"{dg2[2][0]} both expected to make deep runs."
            )
        paras.append(p3)
    else:
        paras.append(
            "The group-stage draw is relatively balanced, with no single group featuring "
            "an overwhelming concentration of favorites. The best-third-place pathway adds "
            "an extra safety net, but finishing second or third still risks a significantly "
            "harder knockout draw."
        )

    conf_sf = defaultdict(list)
    for r in results:
        c = TEAM_CONF.get(r["team"], "Unknown")
        conf_sf[c].append(r["sf_pct"])
    conf_avg = {c: sum(v) / len(v) for c, v in conf_sf.items() if v}
    main_confs = {c: v for c, v in conf_avg.items() if c not in ("OFC", "Unknown")}
    strongest = max(main_confs, key=main_confs.get)
    weakest = min(main_confs, key=main_confs.get)

    paras.append(
        f"At the confederation level, {strongest} leads with an average semifinal reach "
        f"rate of {conf_avg[strongest]:.1f}% per team, reflecting its depth across multiple "
        f"qualifying nations. CONMEBOL teams average {conf_avg.get('CONMEBOL', 0):.1f}%, "
        f"punching above their weight relative to team count. {weakest} teams face the "
        f"steepest climb, averaging {conf_avg[weakest]:.1f}% semifinal probability."
    )

    return "\n".join(f"<p>{p}</p>" for p in paras)


# ── group standings ──────────────────────────────────────────────────────────

def _build_groups(group_standings, results):
    by_team = {r["team"]: r for r in results}
    cards = []
    for g in sorted(group_standings.keys()):
        rows = []
        st = group_standings[g]
        # Sort by advance probability (100 - group_exit_pct)
        for s in st:
            s["adv"] = round(100 - by_team[s["team"]]["group_exit_pct"], 1)
        st.sort(key=lambda x: -x["adv"])

        for i, s in enumerate(st):
            cls = "qual" if i < 2 else ("third" if i == 2 else "out")
            rows.append(
                f'<div class="grp-row {cls}">'
                f'<span class="grp-pos">{i + 1}</span>'
                f'<span class="grp-team">{s["team"]}</span>'
                f'<span class="grp-pct">{s["adv"]:.0f}%</span>'
                f'</div>'
            )
        cards.append(
            f'<div class="grp"><div class="grp-hd">Group {g}</div>{"".join(rows)}</div>'
        )
    return '<div class="groups-grid">' + "".join(cards) + '</div>'


# ── bracket ──────────────────────────────────────────────────────────────────

def _match_card(m):
    """Render one knockout match card."""
    a_win = m["winner"] == m["a"]
    a_cls = " winner" if a_win else ""
    b_cls = "" if a_win else " winner"
    a_pct = m["a_pct"]
    b_pct = round(100 - a_pct, 1)
    return (
        f'<div class="match">'
        f'<div class="tm{a_cls}"><span class="tn">{m["a"]}</span>'
        f'<span class="tp">{a_pct:.0f}%</span></div>'
        f'<div class="tm{b_cls}"><span class="tn">{m["b"]}</span>'
        f'<span class="tp">{b_pct:.0f}%</span></div>'
        f'</div>'
    )


def _build_bracket(bracket):
    sections = []

    sections.append('<h3>Round of 32</h3><div class="bracket-grid r32">')
    for m in bracket["r32"]:
        sections.append(_match_card(m))
    sections.append('</div>')

    sections.append('<h3>Round of 16</h3><div class="bracket-grid r16">')
    for m in bracket["r16"]:
        sections.append(_match_card(m))
    sections.append('</div>')

    sections.append('<h3>Quarterfinals</h3><div class="bracket-grid qf">')
    for m in bracket["qf"]:
        sections.append(_match_card(m))
    sections.append('</div>')

    sections.append('<h3>Semifinals</h3><div class="bracket-grid sf">')
    for m in bracket["sf"]:
        sections.append(_match_card(m))
    sections.append('</div>')

    f = bracket["final"]
    a_win = f["winner"] == f["a"]
    a_cls = " winner" if a_win else ""
    b_cls = "" if a_win else " winner"
    b_pct = round(100 - f["a_pct"], 1)
    sections.append(
        '<h3>Final</h3><div class="final-match">'
        f'<div class="fm-team{a_cls}"><div class="fm-name">{f["a"]}</div>'
        f'<div class="fm-pct">{f["a_pct"]:.0f}%</div></div>'
        f'<div class="fm-vs">vs</div>'
        f'<div class="fm-team{b_cls}"><div class="fm-name">{f["b"]}</div>'
        f'<div class="fm-pct">{b_pct:.0f}%</div></div>'
        f'</div>'
        f'<div class="champion">Predicted Champion: <strong>{f["winner"]}</strong></div>'
    )

    return "\n".join(sections)


# ── full-results table ───────────────────────────────────────────────────────

def _build_table(results):
    rows = []
    for r in results:
        rows.append(
            f"<tr>"
            f"<td>{r['team']}</td>"
            f"<td>{r['win_pct']:.2f}</td>"
            f"<td>{r['final_pct']:.2f}</td>"
            f"<td>{r['sf_pct']:.2f}</td>"
            f"<td>{r['qf_pct']:.2f}</td>"
            f"<td>{r['r32_exit_pct']:.2f}</td>"
            f"<td>{r['group_exit_pct']:.2f}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


# ── styles ───────────────────────────────────────────────────────────────────

CSS = """
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
background:#0f0f1a;color:#e0e0e0;line-height:1.6;padding:2rem;max-width:1200px;margin:0 auto}
h1{font-size:2.2rem;color:#e6a817;text-align:center;margin-bottom:.3rem}
.sub{text-align:center;color:#808099;margin-bottom:2rem;font-size:.95rem}
h2{font-size:1.3rem;color:#e6a817;margin:2rem 0 1rem;border-bottom:1px solid #2a2a4a;padding-bottom:.4rem}
h3{font-size:1.1rem;color:#c0c0d0;margin:1.5rem 0 .75rem}
section{background:#1a1a2e;border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem}
p{margin-bottom:1rem;color:#c0c0d0}
strong{color:#e6a817}

/* groups */
.groups-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.75rem}
@media(max-width:900px){.groups-grid{grid-template-columns:repeat(2,1fr)}}
.grp{background:#16213e;border-radius:8px;padding:.75rem 1rem}
.grp-hd{color:#e6a817;font-weight:700;font-size:.95rem;margin-bottom:.4rem}
.grp-row{display:flex;align-items:center;padding:.25rem 0;font-size:.85rem}
.grp-pos{width:1.2rem;color:#808099;font-weight:600}
.grp-team{flex:1}
.grp-pct{color:#808099;font-size:.8rem;min-width:2.5rem;text-align:right}
.grp-row.qual .grp-team{color:#4ade80}
.grp-row.third .grp-team{color:#facc15}
.grp-row.out{opacity:.45}

/* bracket */
.bracket-grid{display:grid;gap:.6rem}
.r32{grid-template-columns:repeat(4,1fr)}
.r16{grid-template-columns:repeat(4,1fr)}
.qf{grid-template-columns:repeat(2,1fr)}
.sf{grid-template-columns:repeat(2,1fr)}
@media(max-width:700px){.r32,.r16{grid-template-columns:repeat(2,1fr)}}
.match{background:#16213e;border-radius:8px;padding:.5rem .75rem;border-left:3px solid #2a2a4a}
.tm{display:flex;justify-content:space-between;padding:.2rem 0;font-size:.85rem}
.tm.winner .tn{color:#e6a817;font-weight:600}
.tn{flex:1}.tp{color:#808099;font-size:.8rem;min-width:2.5rem;text-align:right}

/* final */
.final-match{display:flex;align-items:center;justify-content:center;gap:1.5rem;
padding:1.5rem;background:#16213e;border-radius:12px;max-width:500px;margin:0 auto}
.fm-team{text-align:center;flex:1}
.fm-name{font-size:1.15rem;font-weight:600;color:#c0c0d0}
.fm-team.winner .fm-name{color:#e6a817;font-size:1.3rem}
.fm-pct{font-size:.9rem;color:#808099;margin-top:.2rem}
.fm-vs{color:#555;font-size:.9rem;font-weight:600}
.champion{text-align:center;margin-top:1rem;font-size:1.1rem;color:#c0c0d0}
.champion strong{font-size:1.25rem}

/* charts */
.chart-box{position:relative;height:480px;margin:1rem 0}

/* table */
.tbl-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.9rem}
th{background:#0d1b2a;padding:.7rem 1rem;text-align:left;cursor:pointer;
user-select:none;position:sticky;top:0;white-space:nowrap}
th:hover{background:#162447}
td{padding:.55rem 1rem;border-bottom:1px solid #0f0f1a}
tr:nth-child(even){background:#16213e}
tr:nth-child(odd){background:#1b2838}
tr:hover{background:#1f3a5f}
.sa{font-size:.75rem;margin-left:3px;opacity:.6}
footer{text-align:center;color:#555;font-size:.8rem;margin-top:2rem}
"""

SORT_JS = """
function sortTbl(c){
 var t=document.getElementById('tbl'),b=t.tBodies[0],
 rs=Array.from(b.rows),h=t.tHead.rows[0].cells[c],
 d=h.dataset.d==='d'?'a':'d';
 t.tHead.querySelectorAll('.sa').forEach(function(s){s.textContent='\\u21C5'});
 h.dataset.d=d;h.querySelector('.sa').textContent=d==='d'?'\\u2193':'\\u2191';
 rs.sort(function(a,b){
  var va=c?parseFloat(a.cells[c].textContent):a.cells[0].textContent,
      vb=c?parseFloat(b.cells[c].textContent):b.cells[0].textContent;
  if(c)return d==='d'?vb-va:va-vb;
  return d==='d'?vb.localeCompare(va):va.localeCompare(vb)});
 rs.forEach(function(r){b.appendChild(r)})
}
"""


# ── assemble ─────────────────────────────────────────────────────────────────

def generate_report():
    data = load_results()
    results = data["results"]
    groups = data["groups"]
    n_iter = data["n_iterations"]
    group_standings = data.get("group_standings", {})
    bracket = data.get("bracket", {})

    analysis = _build_analysis(results, groups)
    groups_html = _build_groups(group_standings, results) if group_standings else ""
    bracket_html = _build_bracket(bracket) if bracket else ""
    table_rows = _build_table(results)

    top16 = results[:16]
    labels_json = json.dumps([r["team"] for r in top16], ensure_ascii=False)
    win_json = json.dumps([r["win_pct"] for r in top16])
    sf_json = json.dumps([r["sf_pct"] for r in top16])

    chart_js = (
        "var lb=" + labels_json + ";\n"
        "var wd=" + win_json + ";\n"
        "var sd=" + sf_json + ";\n"
        "function mkChart(id,data,label,color){\n"
        " new Chart(document.getElementById(id),{\n"
        "  type:'bar',data:{labels:lb,datasets:[{data:data,\n"
        "   backgroundColor:color+'0.75)',borderColor:color+'1)',borderWidth:1,borderRadius:4}]},\n"
        "  options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,\n"
        "   plugins:{legend:{display:false},tooltip:{callbacks:{label:function(c){"
        "return c.raw.toFixed(2)+'%'}}}},\n"
        "   scales:{x:{ticks:{color:'#a0a0b0'},grid:{color:'#2a2a4a'},\n"
        "    title:{display:true,text:label,color:'#a0a0b0'}},\n"
        "    y:{ticks:{color:'#e0e0e0',font:{size:13}},grid:{display:false}}}}\n"
        " })\n"
        "}\n"
        "mkChart('wc',wd,'Win Probability (%)','rgba(230,168,23,');\n"
        "mkChart('sc',sd,'Semifinal Probability (%)','rgba(52,152,219,');\n"
    )

    th_row = (
        '<tr>'
        '<th onclick="sortTbl(0)">Team <span class="sa">⇅</span></th>'
        '<th onclick="sortTbl(1)">Win % <span class="sa">⇅</span></th>'
        '<th onclick="sortTbl(2)">Final % <span class="sa">⇅</span></th>'
        '<th onclick="sortTbl(3)">SF % <span class="sa">⇅</span></th>'
        '<th onclick="sortTbl(4)">QF % <span class="sa">⇅</span></th>'
        '<th onclick="sortTbl(5)">R32 Exit % <span class="sa">⇅</span></th>'
        '<th onclick="sortTbl(6)">Group Exit % <span class="sa">⇅</span></th>'
        '</tr>'
    )

    html = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        '<title>Cup-GPT — World Cup 2026 Predictions</title>\n'
        '<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>\n'
        '<style>' + CSS + '</style>\n'
        '</head>\n<body>\n'
        '<header>\n<h1>Cup-GPT</h1>\n'
        f'<p class="sub">World Cup 2026 Monte Carlo Predictions &mdash; {n_iter:,} simulations</p>\n'
        '</header>\n'
        # Analysis
        '<section>\n<h2>Tournament Analysis</h2>\n' + analysis + '\n</section>\n'
        # Group standings
        + ('<section>\n<h2>Predicted Group Standings</h2>\n'
           '<p style="font-size:.85rem;color:#808099">'
           'Advance probability (%) — '
           '<span style="color:#4ade80">● Qualified</span> '
           '<span style="color:#facc15">● 3rd (best-of)</span> '
           '<span style="opacity:.45">● Eliminated</span>'
           '</p>\n'
           + groups_html + '\n</section>\n' if groups_html else '')
        # Bracket
        + ('<section>\n<h2>Predicted Bracket</h2>\n'
           '<p style="font-size:.85rem;color:#808099">'
           'Most likely path based on group projections and model win probabilities. '
           '<span style="color:#e6a817">Highlighted</span> = predicted winner.'
           '</p>\n'
           + bracket_html + '\n</section>\n' if bracket_html else '')
        # Charts
        + '<section>\n<h2>Win Probability &mdash; Top 16</h2>\n'
        '<div class="chart-box"><canvas id="wc"></canvas></div>\n</section>\n'
        '<section>\n<h2>Semifinal Probability &mdash; Top 16</h2>\n'
        '<div class="chart-box"><canvas id="sc"></canvas></div>\n</section>\n'
        # Table
        '<section>\n<h2>All 48 Teams</h2>\n'
        '<div class="tbl-wrap">\n<table id="tbl">\n'
        '<thead>' + th_row + '</thead>\n'
        '<tbody>\n' + table_rows + '\n</tbody>\n'
        '</table>\n</div>\n</section>\n'
        '<footer>Generated by Cup-GPT &mdash; XGBoost + Monte Carlo simulation</footer>\n'
        '<script>\n' + SORT_JS + '\n' + chart_js + '\n</script>\n'
        '</body>\n</html>'
    )

    out_path = OUTPUT_DIR / "report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    generate_report()
