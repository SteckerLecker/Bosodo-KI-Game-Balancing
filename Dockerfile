# ============================================================
# BOSODO RL-Balancing — Docker Container
# ============================================================

FROM python:3.11-slim

# System-Abhängigkeiten
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Arbeitsverzeichnis
WORKDIR /app

# Python-Abhängigkeiten installieren
# --extra-index-url sorgt dafür, dass torch mit CUDA-Support installiert wird
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
       --extra-index-url https://download.pytorch.org/whl/cu121

# Projektdateien kopieren
COPY . .

# TensorBoard-Port
EXPOSE 6006

# Standardmäßig Training starten
ENTRYPOINT ["python", "scripts/train.py"]
CMD ["--config", "config/training_config.yaml"]
