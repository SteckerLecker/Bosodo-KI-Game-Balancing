import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

# ==========================================
# EINSTELLUNGEN (Defaults – überschreibbar per CLI)
# ==========================================
DEFAULT_INPUT  = '../data/scrum_edition/llm_cache.json'
DEFAULT_OUTPUT = '../data/scrum_edition/llmCacheAuswertung/'
DEFAULT_SCHWELLENWERT = 0.65
# ==========================================

parser = argparse.ArgumentParser(description='Analysiert einen LLM-Cache und erstellt Auswertungsbilder.')
parser.add_argument('--input',  '-i', default=DEFAULT_INPUT,          help=f'Pfad zur llm_cache.json (default: {DEFAULT_INPUT})')
parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT,         help=f'Ausgabeordner für Bilder (default: {DEFAULT_OUTPUT})')
parser.add_argument('--schwellenwert', '-s', type=float, default=DEFAULT_SCHWELLENWERT, help=f'Match-Schwellenwert (default: {DEFAULT_SCHWELLENWERT})')
args = parser.parse_args()

DATEIPFAD      = args.input
OUTPUT_DIR     = args.output
SCHWELLENWERT  = args.schwellenwert

# Erstelle den Ausgabeordner, falls er nicht existiert
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Daten laden und aufbereiten
with open(DATEIPFAD, 'r', encoding='utf-8') as f:
    data = json.load(f)

records = []
for key, value in data.items():
    m, w = key.split('_')
    records.append({'Monster': m, 'Wissenskarte': w, 'Score': value['score']})

df = pd.DataFrame(records)

# 2. Matches berechnen anhand des Schwellenwerts
df['Is_Match'] = df['Score'] >= SCHWELLENWERT

# Matches pro Monster aggregieren
monster_matches = df.groupby('Monster')['Is_Match'].sum().reset_index()
monster_matches.columns = ['Monster', 'Anzahl Matches']
monster_matches = monster_matches.sort_values(by='Anzahl Matches', ascending=False)

# Matches pro Wissenskarte aggregieren
wissen_matches = df.groupby('Wissenskarte')['Is_Match'].sum().reset_index()
wissen_matches.columns = ['Wissenskarte', 'Anzahl Matches']
wissen_matches = wissen_matches.sort_values(by='Anzahl Matches', ascending=False)

# 3. Tabellenausgabe in der Konsole
print(f"=== MATCHES (Schwellenwert >= {SCHWELLENWERT}) ===")
print("\nMonster:\n", monster_matches.to_string(index=False))
print("\nWissenskarten:\n", wissen_matches.to_string(index=False))

# --- BILD 1: Verbindungsmatrix (Heatmap) ---
pivot_table = df.pivot(index='Monster', columns='Wissenskarte', values='Score')

plt.figure(figsize=(14, 10))
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Verbindungs-Score'})
plt.title(f'Verbindungsmatrix (Monster vs. Wissenskarten)\nSchwellenwert für Matches: >= {SCHWELLENWERT}', fontsize=16)
plt.xlabel('Wissenskarten', fontsize=12)
plt.ylabel('Monster', fontsize=12)
plt.tight_layout()

# Speichern im gewünschten Ordner
matrix_path = os.path.join(OUTPUT_DIR, 'verbindungsmatrix.png')
plt.savefig(matrix_path, dpi=150)
plt.close() # Verhindert überlappende Plots

# --- BILD 2: Visuelle Tabellen ---
fig, axes = plt.subplots(1, 2, figsize=(10, 8))

# Linkes Diagramm: Monster-Tabelle
axes[0].axis('off')
axes[0].axis('tight')
table1 = axes[0].table(cellText=monster_matches.values, colLabels=monster_matches.columns, loc='center', cellLoc='center')
table1.auto_set_font_size(False)
table1.set_fontsize(11)
table1.scale(1.2, 1.5)

for key, cell in table1.get_celld().items():
    if key[0] == 0:
        cell.set_facecolor('#4c72b0')
        cell.set_text_props(color='w', weight='bold')
    elif key[0] % 2 == 0:
        cell.set_facecolor('#f2f2f2')

axes[0].set_title(f'Matches pro Monster (>= {SCHWELLENWERT})', fontweight="bold", fontsize=14, pad=20)

# Rechtes Diagramm: Wissenskarten-Tabelle
axes[1].axis('off')
axes[1].axis('tight')
table2 = axes[1].table(cellText=wissen_matches.values, colLabels=wissen_matches.columns, loc='center', cellLoc='center')
table2.auto_set_font_size(False)
table2.set_fontsize(11)
table2.scale(1.2, 1.5)

for key, cell in table2.get_celld().items():
    if key[0] == 0:
        cell.set_facecolor('#55a868')
        cell.set_text_props(color='w', weight='bold')
    elif key[0] % 2 == 0:
        cell.set_facecolor('#f2f2f2')

axes[1].set_title(f'Matches pro Wissenskarte (>= {SCHWELLENWERT})', fontweight="bold", fontsize=14, pad=20)

plt.tight_layout()

# Speichern im gewünschten Ordner
tabellen_path = os.path.join(OUTPUT_DIR, 'match_tabellen.png')
plt.savefig(tabellen_path, bbox_inches='tight', dpi=150)
plt.close()

print(f"\nBilder wurden erfolgreich in '{OUTPUT_DIR}' generiert!")