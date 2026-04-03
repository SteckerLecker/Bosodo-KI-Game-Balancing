# GPU-Server Anleitung — BOSODO RL-Balancing

**Server:** `141.39.193.212` | **Port:** `22122` | **GPUs:** 4x RTX A6000 (Device 4–7)

---

## 1. Voraussetzungen

- SSH-Zugang eingerichtet (public key beim Dozenten hinterlegt)
- Docker lokal installiert
- Projektordner: `/Users/nicoloss/.../bosodo-balancing`

---

## 2. Docker-Image lokal bauen und übertragen

Da der Server **keinen Internetzugang** hat, müssen alle Pakete ins Image eingebaut und lokal gebaut werden.

### Image bauen

```bash
cd /Users/nicoloss/Desktop/Studium/Semester2/KI-Projektarbeit/Workspace_Cardgenerator/bosodo-balancing

# --platform wichtig: lokal macOS, Server läuft auf Linux x86
docker build --platform linux/amd64 -t bosodo-balancing .
docker save bosodo-balancing | gzip > bosodo-balancing.tar.gz
```

### Image auf Server übertragen

```bash
# -P 22122 beachten (Großes P bei scp!)
scp -P 22122 bosodo-balancing.tar.gz ps-stud5@141.39.193.212:~/data/stud5/
```

### Image auf dem Server laden

```bash
ssh ps-stud5@141.39.193.212 -p 22122
docker load < ~/data/stud5/bosodo-balancing.tar.gz
```

---

## 3. Container starten

```bash
# Auf dem Server ausführen
docker run --gpus '"device=4,5,6,7"' \
  -v ~/data/stud5:/proj \
  -w /proj \
  -it --rm \
  --entrypoint bash \
  bosodo-balancing
```

- `--gpus '"device=4,5,6,7"'` — zugewiesene GPUs
- `-v ~/data/stud5:/proj` — persistenter Ordner (überlebt Neustarts)
- `--entrypoint bash` — Shell statt direktem Training-Start

---

## 4. Training ausführen

### Direkt starten

```bash
python /app/scripts/train.py --config /app/config/training_config.yaml
```

### Empfohlen: Mit tmux (überlebt SSH-Trennung)

```bash
# Neue tmux-Session starten
tmux new -s training

# Training starten
python /app/scripts/train.py --config /app/config/training_config.yaml

# tmux loslassen (Training läuft weiter im Hintergrund)
Ctrl+B, dann D

# Später wieder einsteigen
tmux attach -t training
```

---

## 5. Analyse ausführen

```bash
python /app/scripts/analyze.py
```

Ergebnisse werden laut Konfiguration nach `output/reports/` gespeichert.

---

## 6. Ergebnisse sichern

Alles außerhalb von `/proj` geht beim Containerbeenden verloren. Outputs nach `/proj` kopieren:

```bash
cp -r /app/output /proj/
```

Oder direkt in der `training_config.yaml` den Output-Pfad auf `/proj/output/` setzen:

```yaml
output:
  dir: "/proj/output/"

analysis:
  report_path: "/proj/output/reports/"
```

---

## 7. GPU-Auslastung prüfen

```bash
# Einmalig
nvidia-smi

# Live-Monitoring (alle 1 Sekunde)
watch -n 1 nvidia-smi

# PyTorch GPU-Check
python -c "import torch; print(torch.cuda.device_count())"
```

---

## 8. Ergebnisse herunterladen

```bash
# Vom lokalen Rechner ausführen
scp -P 22122 -r ps-stud5@141.39.193.212:~/data/stud5/output ./
```

---

## 9. Schnellreferenz

| Aufgabe | Befehl |
|---|---|
| SSH verbinden | `ssh ps-stud5@141.39.193.212 -p 22122` |
| Container starten | siehe Abschnitt 3 |
| Training starten | `python /app/scripts/train.py --config /app/config/training_config.yaml` |
| Analyse starten | `python /app/scripts/analyze.py` |
| GPU prüfen | `nvidia-smi` |
| tmux starten | `tmux new -s training` |
| tmux anhängen | `tmux attach -t training` |
| Container verlassen | `exit` oder `Ctrl+D` |
| Ergebnisse herunterladen | `scp -P 22122 -r ps-stud5@141.39.193.212:~/data/stud5/output ./` |
