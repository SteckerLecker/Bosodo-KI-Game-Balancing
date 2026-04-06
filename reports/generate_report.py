#!/usr/bin/env python3
"""
Generates a visual HTML balancing report from balancing_report.json.
Usage: python reports/generate_report.py [--input balancing_report.json] [--output reports/report.html]
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def load_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_label(score: float) -> tuple[str, str]:
    """Returns (text, css-class) for a balance score."""
    if score >= 0.8:
        return "Gut", "badge-good"
    elif score >= 0.5:
        return "OK", "badge-ok"
    elif score > 0.0:
        return "Problem", "badge-warn"
    else:
        return "Ausstehend", "badge-neutral"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_html(report: dict) -> str:
    total_ep = report.get("total_episodes", 0)
    avg_len = report.get("avg_game_length", 0)
    avg_def = report.get("avg_defense_rate", 0)
    overall_score = report.get("overall_score", 0)
    issues = report.get("overall_issues", [])

    sym_analysis = report.get("symbol_analysis", {})
    ext_sym = report.get("extended_symbol_metrics", {})
    card_reports = report.get("card_reports", {})
    llm = report.get("llm_analysis", {})

    # --- Symbol data for charts ---
    symbols = list(ext_sym.get("defense_rate_per_symbol", {}).keys())
    def_rates = [ext_sym["defense_rate_per_symbol"].get(s, 0) for s in symbols]
    starv_rates = [ext_sym["symbol_starvation_rate"].get(s, 0) for s in symbols]
    avg_on_hand = [ext_sym["avg_symbol_on_hand"].get(s, 0) for s in symbols]

    # --- Card data ---
    monsters = {k: v for k, v in card_reports.items() if v.get("card_type") == "monster"}
    wisdoms  = {k: v for k, v in card_reports.items() if v.get("card_type") == "wisdom"}

    monster_ids   = [v["card_id"]   for v in monsters.values()]
    monster_names = [v["card_name"] for v in monsters.values()]
    monster_plays = [v["times_played"] for v in monsters.values()]

    wisdom_ids   = [v["card_id"]   for v in wisdoms.values()]
    wisdom_names = [v["card_name"] for v in wisdoms.values()]
    wisdom_plays = [v["times_played"] for v in wisdoms.values()]

    # --- LLM analysis ---
    perfekt   = llm.get("perfekt_count", 0)
    grauzone  = llm.get("grauzone_count", 0)
    fehlz     = llm.get("fehlzuordnung_count", 0)
    total_pairs = llm.get("total_symbol_matching_pairs", 0)
    fehlzuordnungen = llm.get("fehlzuordnungen", [])

    # Sort fehlzuordnungen by score ascending (worst first)
    fehlzuordnungen_sorted = sorted(fehlzuordnungen, key=lambda x: x.get("llm_score", 0))

    # Build issues HTML
    issues_html = ""
    for issue in issues:
        issues_html += f'<li class="issue-item"><span class="issue-icon">⚠</span>{issue}</li>\n'
    if not issues_html:
        issues_html = '<li class="issue-item good"><span class="issue-icon">✓</span>Keine Probleme gefunden.</li>'

    # Build card rows
    def card_rows(cards):
        rows = ""
        for v in cards.values():
            badge_text, badge_class = score_label(v.get("balance_score", 0))
            card_issues = v.get("issues", [])
            issue_str = "; ".join(card_issues) if card_issues else "—"
            rows += f"""<tr>
                <td><code>{v['card_id']}</code></td>
                <td>{v['card_name']}</td>
                <td>{v['times_played']}</td>
                <td><span class="badge {badge_class}">{badge_text}</span></td>
                <td class="issue-cell">{issue_str}</td>
            </tr>\n"""
        return rows

    # Build fehlzuordnung rows (top 20 worst)
    fehl_rows = ""
    for f in fehlzuordnungen_sorted:
        score = f.get("llm_score", 0)
        color_class = "score-bad" if score <= 0.2 else "score-mid"
        symbols_str = ", ".join(f.get("shared_symbols", []))
        fehl_rows += f"""<tr>
            <td><code>{f['monster_id']}</code> {f['monster_name']}</td>
            <td><code>{f['wisdom_id']}</code> {f['wisdom_name']}</td>
            <td>{symbols_str}</td>
            <td><span class="{color_class}">{score:.1f}</span></td>
            <td>{f.get('begruendung', '')}</td>
        </tr>\n"""

    generated_at = datetime.now().strftime("%d.%m.%Y %H:%M")

    overall_pct = int(overall_score * 100)

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Bosodo Balancing Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #222535;
    --border: #2e3147;
    --accent: #6c63ff;
    --accent2: #ff6584;
    --green: #43d98b;
    --yellow: #f9c74f;
    --red: #f94144;
    --text: #e2e8f0;
    --muted: #8892a4;
    --radius: 12px;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; line-height: 1.6; }}
  a {{ color: var(--accent); text-decoration: none; }}

  /* Layout */
  .container {{ max-width: 1200px; margin: 0 auto; padding: 24px 20px; }}
  header {{ padding: 32px 0 24px; border-bottom: 1px solid var(--border); margin-bottom: 32px; display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 12px; }}
  header h1 {{ font-size: 28px; font-weight: 700; letter-spacing: -0.5px; }}
  header h1 span {{ color: var(--accent); }}
  .meta {{ color: var(--muted); font-size: 12px; }}

  /* Grid */
  .grid {{ display: grid; gap: 20px; }}
  .grid-4 {{ grid-template-columns: repeat(4, 1fr); }}
  .grid-2 {{ grid-template-columns: repeat(2, 1fr); }}
  .grid-3 {{ grid-template-columns: repeat(3, 1fr); }}
  @media (max-width: 900px) {{ .grid-4, .grid-3 {{ grid-template-columns: repeat(2,1fr); }} }}
  @media (max-width: 600px) {{ .grid-4, .grid-3, .grid-2 {{ grid-template-columns: 1fr; }} }}

  /* Section */
  .section {{ margin-bottom: 36px; }}
  .section-title {{ font-size: 16px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }}
  .section-title::after {{ content: ''; flex: 1; height: 1px; background: var(--border); }}

  /* Card */
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; }}
  .card-sm {{ padding: 16px; }}

  /* Stat tile */
  .stat-tile {{ display: flex; flex-direction: column; gap: 6px; }}
  .stat-tile .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: var(--muted); }}
  .stat-tile .value {{ font-size: 32px; font-weight: 700; line-height: 1; }}
  .stat-tile .sub {{ font-size: 12px; color: var(--muted); }}
  .value-green {{ color: var(--green); }}
  .value-yellow {{ color: var(--yellow); }}
  .value-red {{ color: var(--red); }}
  .value-accent {{ color: var(--accent); }}

  /* Score ring */
  .score-ring-wrap {{ display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }}
  .ring-canvas {{ position: relative; width: 100px; height: 100px; flex-shrink: 0; }}
  .ring-canvas canvas {{ position: absolute; top: 0; left: 0; }}
  .ring-label {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 20px; font-weight: 700; color: var(--text); text-align: center; pointer-events: none; }}

  /* Issues */
  .issue-list {{ list-style: none; display: flex; flex-direction: column; gap: 8px; }}
  .issue-item {{ display: flex; align-items: flex-start; gap: 10px; padding: 10px 14px; background: var(--surface2); border-radius: 8px; border-left: 3px solid var(--yellow); font-size: 13px; }}
  .issue-item.good {{ border-color: var(--green); }}
  .issue-icon {{ font-size: 15px; flex-shrink: 0; margin-top: 1px; }}

  /* Symbol metrics */
  .sym-grid {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; }}
  @media (max-width: 600px) {{ .sym-grid {{ grid-template-columns: 1fr; }} }}
  .sym-card {{ background: var(--surface2); border-radius: 10px; padding: 14px 16px; border: 1px solid var(--border); }}
  .sym-name {{ font-size: 20px; font-weight: 700; color: var(--accent); margin-bottom: 10px; }}
  .sym-row {{ display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; }}
  .sym-row .lbl {{ color: var(--muted); }}
  .bar-wrap {{ background: var(--border); border-radius: 4px; height: 6px; margin-top: 2px; margin-bottom: 6px; }}
  .bar-fill {{ height: 6px; border-radius: 4px; }}
  .bar-green {{ background: var(--green); }}
  .bar-yellow {{ background: var(--yellow); }}
  .bar-red {{ background: var(--red); }}

  /* Chart containers */
  .chart-wrap {{ position: relative; height: 260px; }}
  .chart-wrap-sm {{ position: relative; height: 200px; }}

  /* Tables */
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  thead tr {{ border-bottom: 2px solid var(--border); }}
  th {{ text-align: left; padding: 8px 10px; color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: var(--surface2); }}
  code {{ background: var(--surface2); padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 12px; color: var(--accent); }}
  .issue-cell {{ color: var(--muted); font-size: 12px; max-width: 260px; }}

  /* Badges */
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }}
  .badge-good    {{ background: rgba(67,217,139,.15); color: var(--green); }}
  .badge-ok      {{ background: rgba(249,199,79,.15); color: var(--yellow); }}
  .badge-warn    {{ background: rgba(249,65,68,.15);  color: var(--red); }}
  .badge-neutral {{ background: rgba(255,255,255,.07); color: var(--muted); }}

  /* LLM scores */
  .score-bad {{ color: var(--red); font-weight: 700; }}
  .score-mid {{ color: var(--yellow); font-weight: 600; }}

  /* LLM summary tiles */
  .llm-tiles {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin-bottom: 20px; }}
  @media (max-width: 600px) {{ .llm-tiles {{ grid-template-columns: 1fr; }} }}
  .llm-tile {{ background: var(--surface2); border-radius: 10px; padding: 14px 16px; text-align: center; border: 1px solid var(--border); }}
  .llm-tile .tval {{ font-size: 28px; font-weight: 700; }}
  .llm-tile .tlbl {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.6px; margin-top: 4px; }}
  .t-green {{ color: var(--green); }}
  .t-yellow {{ color: var(--yellow); }}
  .t-red {{ color: var(--red); }}

  /* Tabs */
  .tabs {{ display: flex; gap: 4px; margin-bottom: 16px; border-bottom: 1px solid var(--border); padding-bottom: 0; }}
  .tab-btn {{ background: none; border: none; color: var(--muted); cursor: pointer; padding: 8px 16px; font-size: 13px; font-weight: 500; border-bottom: 2px solid transparent; margin-bottom: -1px; transition: all .15s; }}
  .tab-btn.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
  .tab-btn:hover {{ color: var(--text); }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}

  footer {{ margin-top: 48px; padding-top: 20px; border-top: 1px solid var(--border); color: var(--muted); font-size: 12px; text-align: center; }}
  .table-scroll {{ max-height: 520px; overflow-y: auto; border-radius: 8px; }}
  .table-scroll table {{ min-width: 700px; }}
  .table-scroll thead tr {{ position: sticky; top: 0; background: var(--surface); z-index: 1; }}
</style>
</head>
<body>
<div class="container">

  <header>
    <div>
      <h1>Bosodo <span>Balancing</span> Report</h1>
      <div class="meta">Generiert am {generated_at} &nbsp;·&nbsp; {total_ep} Episoden simuliert</div>
    </div>
  </header>

  <!-- KPIs -->
  <div class="section">
    <div class="section-title">Übersicht</div>
    <div class="grid grid-4">
      <div class="card stat-tile">
        <div class="label">Episoden</div>
        <div class="value value-accent">{total_ep:,}</div>
        <div class="sub">Simulierte Spielrunden</div>
      </div>
      <div class="card stat-tile">
        <div class="label">Ø Spiellänge</div>
        <div class="value {'value-red' if avg_len < 8 else 'value-green'}">{avg_len:.1f}</div>
        <div class="sub">Züge pro Spiel (Ziel ≥ 8)</div>
      </div>
      <div class="card stat-tile">
        <div class="label">Ø Abwehrrate</div>
        <div class="value {'value-green' if avg_def >= 0.6 else 'value-yellow'}">{pct(avg_def)}</div>
        <div class="sub">Monster erfolgreich abgewehrt</div>
      </div>
      <div class="card stat-tile">
        <div class="label">Multi-Symbol-Abwehr</div>
        <div class="value value-accent">{pct(ext_sym.get('multi_symbol_defense_rate', 0))}</div>
        <div class="sub">Abwehr mit mehreren Symbolen</div>
      </div>
    </div>
  </div>

  <!-- Issues -->
  <div class="section">
    <div class="section-title">Probleme & Hinweise</div>
    <div class="card">
      <ul class="issue-list">
        {issues_html}
      </ul>
    </div>
  </div>

  <!-- Symbol Metrics -->
  <div class="section">
    <div class="section-title">Symbol-Analyse</div>
    <div class="sym-grid">
      {"".join([f'''
      <div class="sym-card">
        <div class="sym-name">{s}</div>
        <div class="sym-row"><span class="lbl">Abwehrrate</span><span>{pct(ext_sym["defense_rate_per_symbol"].get(s, 0))}</span></div>
        <div class="bar-wrap"><div class="bar-fill {'bar-green' if ext_sym['defense_rate_per_symbol'].get(s,0)>=0.6 else 'bar-yellow'}" style="width:{min(ext_sym['defense_rate_per_symbol'].get(s,0)*100,100):.0f}%"></div></div>
        <div class="sym-row"><span class="lbl">Starvation</span><span class="{'t-red' if ext_sym['symbol_starvation_rate'].get(s,0)>0.15 else 't-green'}">{pct(ext_sym['symbol_starvation_rate'].get(s, 0))}</span></div>
        <div class="bar-wrap"><div class="bar-fill {'bar-red' if ext_sym['symbol_starvation_rate'].get(s,0)>0.15 else 'bar-green'}" style="width:{min(ext_sym['symbol_starvation_rate'].get(s,0)*100*2,100):.0f}%"></div></div>
        <div class="sym-row"><span class="lbl">Ø auf der Hand</span><span class="{'t-red' if ext_sym['avg_symbol_on_hand'].get(s,0)<1.5 else 't-green'}">{ext_sym['avg_symbol_on_hand'].get(s, 0):.2f}</span></div>
        <div class="bar-wrap"><div class="bar-fill {'bar-red' if ext_sym['avg_symbol_on_hand'].get(s,0)<1.5 else 'bar-green'}" style="width:{min(ext_sym['avg_symbol_on_hand'].get(s,0)/3*100,100):.0f}%"></div></div>
        <div class="sym-row"><span class="lbl">Monster / Wissenskarten</span><span>{sym_analysis.get('symbol_analysis', {}).get('coverage', {}).get(s, {}).get('monster_count', sym_analysis.get('coverage', {}).get(s, {}).get('monster_count','?'))} / {sym_analysis.get('symbol_analysis', {}).get('coverage', {}).get(s, {}).get('wisdom_count', sym_analysis.get('coverage', {}).get(s, {}).get('wisdom_count','?'))}</span></div>
      </div>''' for s in symbols])}
    </div>
  </div>

  <!-- Charts Row -->
  <div class="section">
    <div class="section-title">Charts</div>
    <div class="grid grid-3">
      <div class="card">
        <div style="font-size:13px;font-weight:600;margin-bottom:12px;color:var(--muted)">Abwehrrate je Symbol</div>
        <div class="chart-wrap"><canvas id="chartDefRate"></canvas></div>
      </div>
      <div class="card">
        <div style="font-size:13px;font-weight:600;margin-bottom:12px;color:var(--muted)">Starvation Rate je Symbol</div>
        <div class="chart-wrap"><canvas id="chartStarv"></canvas></div>
      </div>
      <div class="card">
        <div style="font-size:13px;font-weight:600;margin-bottom:12px;color:var(--muted)">LLM Matching-Qualität</div>
        <div class="chart-wrap"><canvas id="chartLLM"></canvas></div>
      </div>
    </div>
  </div>

  <!-- Cards Section with Tabs -->
  <div class="section">
    <div class="section-title">Karten-Statistiken</div>
    <div class="card">
      <div class="tabs">
        <button class="tab-btn active" onclick="switchTab('tab-monster', this)">Monster ({len(monsters)})</button>
        <button class="tab-btn" onclick="switchTab('tab-wisdom', this)">Wissen ({len(wisdoms)})</button>
      </div>

      <div id="tab-monster" class="tab-panel active">
        <div class="chart-wrap" style="height:220px;margin-bottom:20px"><canvas id="chartMonsterPlays"></canvas></div>
        <div class="table-scroll"><table>
          <thead><tr><th>ID</th><th>Name</th><th>Gespielt</th><th>Status</th><th>Probleme</th></tr></thead>
          <tbody>{card_rows(monsters)}</tbody>
        </table></div>
      </div>

      <div id="tab-wisdom" class="tab-panel">
        <div class="chart-wrap" style="height:220px;margin-bottom:20px"><canvas id="chartWisdomPlays"></canvas></div>
        <div class="table-scroll"><table>
          <thead><tr><th>ID</th><th>Name</th><th>Gespielt</th><th>Status</th><th>Probleme</th></tr></thead>
          <tbody>{card_rows(wisdoms)}</tbody>
        </table></div>
      </div>
    </div>
  </div>

  <!-- LLM Analysis -->
  <div class="section">
    <div class="section-title">LLM Themen-Matching</div>
    <div class="llm-tiles">
      <div class="llm-tile">
        <div class="tval t-green">{perfekt}</div>
        <div class="tlbl">Perfekte Matches</div>
      </div>
      <div class="llm-tile">
        <div class="tval t-yellow">{grauzone}</div>
        <div class="tlbl">Grauzone</div>
      </div>
      <div class="llm-tile">
        <div class="tval t-red">{fehlz}</div>
        <div class="tlbl">Fehlzuordnungen</div>
      </div>
    </div>
    <div class="card">
      <div style="font-size:13px;color:var(--muted);margin-bottom:14px">
        Alle {len(fehlzuordnungen_sorted)} Fehlzuordnungen (von {total_pairs} Paaren insgesamt) · sortiert nach Score aufsteigend
      </div>
      <div class="table-scroll"><table>
        <thead><tr><th>Monster</th><th>Wissenskarte</th><th>Symbol</th><th>Score</th><th>Begründung</th></tr></thead>
        <tbody>{fehl_rows}</tbody>
      </table></div>
    </div>
  </div>

  <footer>Bosodo Balancing Report &nbsp;·&nbsp; Generiert am {generated_at}</footer>
</div>

<script>
const ACCENT  = '#6c63ff';
const GREEN   = '#43d98b';
const YELLOW  = '#f9c74f';
const RED     = '#f94144';
const MUTED   = '#8892a4';
const SURFACE = '#222535';
const BORDER  = '#2e3147';

Chart.defaults.color = MUTED;
Chart.defaults.borderColor = BORDER;

// --- Defense Rate ---
new Chart(document.getElementById('chartDefRate'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(symbols)},
    datasets: [{{
      label: 'Abwehrrate',
      data: {json.dumps([round(d,3) for d in def_rates])},
      backgroundColor: {json.dumps(['rgba(67,217,139,.7)' if d >= 0.6 else 'rgba(249,199,79,.7)' for d in def_rates])},
      borderRadius: 6,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    scales: {{
      y: {{ min: 0, max: 1, ticks: {{ callback: v => (v*100).toFixed(0)+'%' }} }},
      x: {{ grid: {{ display: false }} }}
    }},
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: ctx => (ctx.parsed.y*100).toFixed(1)+'%' }} }}
    }}
  }}
}});

// --- Starvation Rate ---
new Chart(document.getElementById('chartStarv'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(symbols)},
    datasets: [{{
      label: 'Starvation',
      data: {json.dumps([round(s,3) for s in starv_rates])},
      backgroundColor: {json.dumps(['rgba(249,65,68,.7)' if s > 0.15 else 'rgba(67,217,139,.7)' for s in starv_rates])},
      borderRadius: 6,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    scales: {{
      y: {{ min: 0, max: 0.5, ticks: {{ callback: v => (v*100).toFixed(0)+'%' }} }},
      x: {{ grid: {{ display: false }} }}
    }},
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: ctx => (ctx.parsed.y*100).toFixed(1)+'%' }} }}
    }}
  }}
}});

// --- LLM Donut ---
new Chart(document.getElementById('chartLLM'), {{
  type: 'doughnut',
  data: {{
    labels: ['Perfekt', 'Grauzone', 'Fehlzuordnung'],
    datasets: [{{
      data: [{perfekt}, {grauzone}, {fehlz}],
      backgroundColor: ['rgba(67,217,139,.8)', 'rgba(249,199,79,.8)', 'rgba(249,65,68,.8)'],
      borderColor: SURFACE,
      borderWidth: 3,
      hoverOffset: 8,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ padding: 16, font: {{ size: 12 }} }} }},
      tooltip: {{ callbacks: {{ label: ctx => ctx.label + ': ' + ctx.parsed + ' (' + (ctx.parsed/{total_pairs}*100).toFixed(1)+'%)' }} }}
    }},
    cutout: '65%',
  }}
}});

// --- Monster plays ---
new Chart(document.getElementById('chartMonsterPlays'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(monster_ids)},
    datasets: [{{
      label: 'Gespielt',
      data: {json.dumps(monster_plays)},
      backgroundColor: 'rgba(108,99,255,.7)',
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    scales: {{
      y: {{ beginAtZero: true }},
      x: {{ grid: {{ display: false }} }}
    }},
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ title: (items) => {json.dumps(dict(zip(monster_ids, monster_names)))}[items[0].label] || items[0].label }} }}
    }}
  }}
}});

// --- Wisdom plays ---
new Chart(document.getElementById('chartWisdomPlays'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(wisdom_ids)},
    datasets: [{{
      label: 'Gespielt',
      data: {json.dumps(wisdom_plays)},
      backgroundColor: 'rgba(255,101,132,.7)',
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    scales: {{
      y: {{ beginAtZero: true }},
      x: {{ grid: {{ display: false }} }}
    }},
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ title: (items) => {json.dumps(dict(zip(wisdom_ids, wisdom_names)))}[items[0].label] || items[0].label }} }}
    }}
  }}
}});

// --- Tab switching ---
function switchTab(id, btn) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}}
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate HTML balancing report")
    parser.add_argument("--input",  default="balancing_report.json", help="Path to balancing_report.json")
    parser.add_argument("--output", default="report.html",   help="Output HTML file")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Lade Report: {input_path}")
    report = load_report(input_path)

    print("Generiere HTML...")
    html = render_html(report)

    output_path.write_text(html, encoding="utf-8")
    print(f"Report gespeichert: {output_path}")


if __name__ == "__main__":
    main()
