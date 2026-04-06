#!/bin/bash
set -e

# ============================================================
# BOSODO — Deploy Script
# Baut das Docker-Image lokal und überträgt es auf den GPU-Server
# ============================================================

SERVER="141.39.193.212"
PORT="22122"
USER="ps-stud5"
REMOTE_DIR="~/data/stud5"
IMAGE_NAME="bosodo-balancing"
TAR_FILE="${IMAGE_NAME}.tar.gz"

echo "==> [1/4] Docker-Image bauen..."
docker build --platform linux/amd64 -t "$IMAGE_NAME" .

echo "==> [2/4] Image exportieren..."
docker save "$IMAGE_NAME" | gzip > "$TAR_FILE"
echo "    Größe: $(du -sh "$TAR_FILE" | cut -f1)"

echo "==> [3/4] Image auf Server übertragen..."
scp -P "$PORT" "$TAR_FILE" "${USER}@${SERVER}:${REMOTE_DIR}/"

echo "==> [4/4] Image auf Server laden..."
ssh "${USER}@${SERVER}" -p "$PORT" "docker load < ${REMOTE_DIR}/${TAR_FILE}"

echo ""
echo "============================================================"
echo "Fertig! So geht's weiter:"
echo "============================================================"
echo ""
echo "1) SSH auf den Server:"
echo "   ssh ps-stud5@141.39.193.212 -p 22122"
echo ""
echo "2) Container starten:"
echo "   docker run --gpus '\"device=4,5,6,7\"' -v ~/data/stud5:/proj -w /proj -it --rm --entrypoint bash ${IMAGE_NAME}"
echo ""
echo "3) Iteratives Balancing ausfuehren (im Container):"
echo "   python /app/scripts/iterative_balance.py --work-dir /proj/bosodo-balancing"
echo "   (WICHTIG: --work-dir /proj sorgt dafuer, dass Output auf dem gemounteten Volume landet!)"
echo ""
echo "   Optionen:"
echo "   python /app/scripts/iterative_balance.py --work-dir /proj --timesteps 1000000 --max-iterations 15"
echo "   python /app/scripts/iterative_balance.py --work-dir /proj --start-version 3   # Ab Version 3 weitermachen"
echo ""
echo "   Alternativ manuell trainieren/analysieren:"
echo "   python /app/scripts/train.py --config /app/config/training_config.yaml"
echo "   python /app/scripts/analyze.py"
echo ""
echo "4) Ergebnisse herunterladen (lokal auf deinem Mac, nicht im Container):"
echo "   scp -P 22122 ps-stud5@141.39.193.212:~/data/stud5/bosodo-balancing/output/iterative_balancing/summary.json ./results/"
echo "   scp -r -P 22122 ps-stud5@141.39.193.212:~/data/stud5/bosodo-balancing/output/iterative_balancing/ ./results/iterative_balancing/"
echo "   scp -r -P 22122 "ps-stud5@141.39.193.212:~/data/stud5/bosodo-balancing/data_v*" ./results/iterative_balancing"
echo ""
echo "   Alternativ manuell trainieren/analysieren:"
echo "   scp -P 22122 ps-stud5@141.39.193.212:~/data/stud5/bosodo-balancing/output/reports/balancing_report.json ./reports/"
echo "============================================================"