#!/usr/bin/env bash
# Job diario no HOST: avalia drift, e-mail, retreino condicionado, versiona artefatos.
#
# O container nao commita: credencial de git nao entra na imagem.
#
# Uso:   ./retreinar.sh
# Env:   APP_PORT (padrao 8005)  REPO (padrao /srv/Sistematizacao)
set -euo pipefail

PORTA="${APP_PORT:-8005}"
REPO="${REPO:-/srv/Sistematizacao}"
BASE="http://localhost:${PORTA}"

# 1) monitora (e so retreina se houver drift). Optuna pode levar varios minutos.
curl -fsS --max-time 3600 -X POST "${BASE}/modelo/monitorar?retreinar=true"
echo

# 2) versiona os artefatos no git do host (no-op se nao houve run novo)
cd "$REPO"
git add python/modelos
if git diff --cached --quiet; then
  echo "Nada novo em python/modelos para commitar."
else
  git commit -m "modelo: monitor/retrain $(date -u +%FT%TZ)"
  git push
fi

# 3) reload a quente (seguro mesmo quando nao houve promocao)
curl -fsS -X POST "${BASE}/modelo/recarregar"
echo
