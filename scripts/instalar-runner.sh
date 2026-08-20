#!/usr/bin/env bash
# Instala o self-hosted runner deste repositorio no home server e o deixa
# subindo automaticamente no boot (servico systemd).
#
# Uso:
#   ./instalar-runner.sh <TOKEN_DE_REGISTRO>
#
# O token sai de: GitHub -> Settings -> Actions -> Runners -> New self-hosted
# runner. Ele expira em ~1 hora; se der erro de token, pegue outro na pagina.
#
# Idempotente: pode rodar de novo para re-registrar (usa --replace).

set -euo pipefail

TOKEN="${1:-}"
if [ -z "$TOKEN" ]; then
  echo "ERRO: falta o token de registro."
  echo "Uso: $0 <TOKEN_DE_REGISTRO>"
  exit 1
fi

REPO_URL="https://github.com/luisrobertolrm/sistematizacao"
VERSAO="2.336.0"
SHA256="04cf0be1aff4c3ec3554466c39124ca250e3effd8873bb7e8d68535aa9505d5d"

DIR_RUNNER="/DATA/actions-runner/sistematizacao"
DIR_APP="/DATA/sistematizacao"
APP_PORT="${APP_PORT:-8005}"

TARBALL="actions-runner-linux-x64-${VERSAO}.tar.gz"

echo "==> 1/6 Pre-requisitos"
command -v rsync >/dev/null || sudo apt-get install -y rsync
command -v docker >/dev/null || { echo "ERRO: docker nao encontrado."; exit 1; }
mkdir -p "$DIR_APP" "$DIR_RUNNER"

echo "==> 2/6 .env do app (porta publicada no host)"
if [ -f "$DIR_APP/.env" ]; then
  echo "    ja existe, preservando: $(cat "$DIR_APP/.env")"
else
  echo "APP_PORT=${APP_PORT}" > "$DIR_APP/.env"
  echo "    criado com APP_PORT=${APP_PORT}"
fi

echo "==> 3/6 Download do runner ${VERSAO}"
cd "$DIR_RUNNER"
if [ -f "./config.sh" ]; then
  echo "    runner ja extraido, pulando download"
else
  curl -fSL -o "$TARBALL" \
    "https://github.com/actions/runner/releases/download/v${VERSAO}/${TARBALL}"
  echo "${SHA256}  ${TARBALL}" | shasum -a 256 -c
  tar xzf "./${TARBALL}"
  rm -f "./${TARBALL}"
fi

echo "==> 4/6 Registrando no repositorio"
# --unattended: sem perguntas interativas, entao nenhum label customizado e
# adicionado por engano (o deploy.yml usa `runs-on: self-hosted`).
if [ -f ".runner" ]; then
  sudo ./svc.sh stop 2>/dev/null || true
  sudo ./svc.sh uninstall 2>/dev/null || true
  ./config.sh remove --token "$TOKEN" 2>/dev/null || true
fi
./config.sh \
  --unattended \
  --replace \
  --url "$REPO_URL" \
  --token "$TOKEN" \
  --name "casa-sistematizacao" \
  --work "_work"

echo "==> 5/6 Instalando o servico (auto-start no boot)"
sudo ./svc.sh install "$USER"
sudo ./svc.sh start

echo "==> 6/6 Verificacao"
sleep 3
SERVICO="$(systemctl list-units --type=service --all --no-legend \
  | grep -o 'actions\.runner\.[^ ]*sistematizacao[^ ]*\.service' | head -1)"

if [ -z "$SERVICO" ]; then
  echo "AVISO: nao localizei a unit do servico. Rode: sudo ./svc.sh status"
  exit 1
fi

echo "    unit:    $SERVICO"
echo "    ativo:   $(systemctl is-active "$SERVICO")"
echo "    no boot: $(systemctl is-enabled "$SERVICO")"
echo
journalctl -u "$SERVICO" -n 15 --no-pager | grep -E "Connected to GitHub|Listening for Jobs" \
  || echo "    (ainda conectando; veja: journalctl -u $SERVICO -f)"

echo
echo "Pronto. O runner deve aparecer como Idle em:"
echo "  ${REPO_URL}/settings/actions/runners"
