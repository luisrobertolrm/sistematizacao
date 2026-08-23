#!/usr/bin/env bash
# Instala ou atualiza retreino.service + retreino.timer no systemd do host.
# Rode uma vez com sudo (ou deixe o deploy chamar se NOPASSWD estiver ok):
#   sudo /DATA/sistematizacao/scripts/instalar-timer.sh
set -euo pipefail

REPO="${REPO:-/DATA/sistematizacao}"
chmod +x "${REPO}/scripts/retreinar.sh"
cp "${REPO}/scripts/retreino.service" /etc/systemd/system/retreino.service
cp "${REPO}/scripts/retreino.timer" /etc/systemd/system/retreino.timer
systemctl daemon-reload
systemctl enable --now retreino.timer
systemctl is-enabled retreino.timer
systemctl list-timers retreino.timer --no-pager
