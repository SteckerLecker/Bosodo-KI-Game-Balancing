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
echo "Fertig! Container starten mit:"
echo ""
echo "ssh ps-stud5@141.39.193.212 -p 22122"
echo ""
echo "docker run --gpus '\"device=4,5,6,7\"' -v ~/data/stud5:/proj -w /proj -it --rm --entrypoint bash ${IMAGE_NAME}"
echo ""
echo "python /app/scripts/train.py --config /app/config/training_config.yaml"
echo ""
echo "python /app/scripts/analyze.py"
echo ""
echo "Download balancing_report.json:"
echo "python /app/scripts/analyze.py"
