#!/usr/bin/env bash
# Re-treino do modelo de churn, disparado no HOST (nunca dentro do container).
#
# O container nao commita: credencial de git nao entra na imagem. Quem dispara
# o treino versiona os artefatos depois, pela pasta bind-montada em
# python/modelos.
#
# Uso:   ./retreinar.sh
# Env:   APP_PORT (padrao 8005)  REPO (padrao /srv/Sistematizacao)
set -euo pipefail

PORTA="${APP_PORT:-8005}"
REPO="${REPO:-/srv/Sistematizacao}"
BASE="http://localhost:${PORTA}"

# 1) treina (gera o run versionado) e promove automaticamente
curl -fsS -X POST "${BASE}/modelo/treinar?incluir_novos=true&promover=true"
echo

# 2) versiona os artefatos no git do host
cd "$REPO"
git add python/modelos
if git diff --cached --quiet; then
  echo "Nada novo em python/modelos para commitar."
else
  git commit -m "modelo: retrain $(date -u +%FT%TZ)"
  git push
fi

# 3) garante o reload a quente (o endpoint ja recarrega quando promover=true)
curl -fsS -X POST "${BASE}/modelo/recarregar"
echo
