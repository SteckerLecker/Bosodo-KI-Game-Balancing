# Neuer Container mit NVIDIA PyTorch Image

Alternative zum selbst gebauten `bosodo-balancing`-Image. Nutzt das offizielle NVIDIA PyTorch Image mit Python 3.10+ und aktuellem CUDA-Support.

**Voraussetzung:** Das Projekt liegt bereits auf dem Server unter `~/data/stud5/bosodo-balancing`.

---

## 1. SSH auf den Server

```bash
ssh ps-stud5@141.39.193.212 -p 22122
```

---

## 2. Container starten

```bash
docker run --gpus '"device=4,5,6,7"' -v ~/data/stud5:/proj -w /proj/bosodo-balancing -it --rm nvcr.io/nvidia/pytorch:24.10-py3 bash
```

- `--gpus '"device=4,5,6,7"'` — zugewiesene GPUs
- `-v ~/data/stud5:/proj` — persistenter Ordner (überlebt Container-Neustarts)
- `-w /proj/bosodo-balancing` — startet direkt im Projektordner
- Das Image wird beim ersten Start heruntergeladen (~15-20 GB)

---

## 3. Dependencies installieren

```bash
pip install -r requirements.txt
pip install -e .
```

> **Hinweis:** Der Server hat keinen Internetzugang. Falls `pip install` fehlschlägt, müssen die Pakete vorher lokal heruntergeladen und auf den Server kopiert werden (siehe Abschnitt 6).

---

## 4. Training ausführen

### Iteratives Balancing (empfohlen)

```bash
python scripts/iterative_balance.py --work-dir /proj/bosodo-balancing
```

**Optionen:**

```bash
# Mit angepassten Parametern
python scripts/iterative_balance.py --work-dir /proj/bosodo-balancing --timesteps 1000000 --max-iterations 15

# Ab bestimmter Version weitermachen
python scripts/iterative_balance.py --work-dir /proj/bosodo-balancing --start-version 3
```

### Manuelles Training / Analyse

```bash
python scripts/train.py --config config/training_config.yaml
python scripts/analyze.py
```

### Mit tmux (überlebt SSH-Trennung)

```bash
# Neue tmux-Session starten
tmux new -s training

# Training starten
python scripts/iterative_balance.py --work-dir /proj/bosodo-balancing

# tmux loslassen: Ctrl+B, dann D
# Später wieder einsteigen:
tmux attach -t training
```

---

## 5. Ergebnisse herunterladen

Vom lokalen Rechner (nicht im Container):

```bash
# Iterative Balancing Ergebnisse
scp -r -P 22122 ps-stud5@141.39.193.212:~/data/stud5/bosodo-balancing/output/iterative_balancing/ ./results/iterative_balancing/

# Zusammenfassung
scp -P 22122 ps-stud5@141.39.193.212:~/data/stud5/bosodo-balancing/output/iterative_balancing/summary.json ./results/iterative_balancing/

# Kartendaten aller Versionen
scp -r -P 22122 "ps-stud5@141.39.193.212:~/data/stud5/bosodo-balancing/data_v*" ./results/iterative_balancing/
```

---

## 6. Offline-Installation (falls kein Internet im Container)

Falls der Container keinen Internetzugang hat, Pakete lokal herunterladen und übertragen:

```bash
# Lokal auf dem Mac
mkdir /tmp/bosodo-wheels
pip download -r requirements.txt -d /tmp/bosodo-wheels --platform manylinux2014_x86_64 --python-version 310 --only-binary=:all:
scp -r -P 22122 /tmp/bosodo-wheels ps-stud5@141.39.193.212:~/data/stud5/bosodo-wheels

# Im Container
pip install --no-index --find-links /proj/bosodo-wheels -r requirements.txt
pip install -e .
```

---

## 7. Schnellreferenz

| Aufgabe | Befehl |
|---|---|
| SSH verbinden | `ssh ps-stud5@141.39.193.212 -p 22122` |
| Container starten | siehe Abschnitt 2 |
| Dependencies installieren | `pip install -r requirements.txt && pip install -e .` |
| Iteratives Balancing | `python scripts/iterative_balance.py --work-dir /proj/bosodo-balancing` |
| Training manuell | `python scripts/train.py --config config/training_config.yaml` |
| Analyse manuell | `python scripts/analyze.py` |
| GPU prüfen | `nvidia-smi` |
| Python Version prüfen | `python --version` |
| tmux starten | `tmux new -s training` |
| tmux anhängen | `tmux attach -t training` |
| Container verlassen | `exit` oder `Ctrl+D` |
