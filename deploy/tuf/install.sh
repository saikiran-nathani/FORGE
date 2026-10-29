#!/usr/bin/env bash
# Install or update the FORGE API on the TUF. Run it ON the TUF:
#
#     bash ~/projects/FORGE/deploy/tuf/install.sh            # install / update, stay private
#     bash ~/projects/FORGE/deploy/tuf/install.sh --funnel   # ...and expose it publicly on :8443
#
# First time (repo not cloned yet):
#     git clone git@github.com:saikiran-nathani/FORGE.git ~/projects/FORGE
#
# Safe to re-run: it pulls the latest pushed code, syncs dependencies, reinstalls the
# unit and restarts the service. It never touches the Platform hub or its port (443).
set -euo pipefail

APP="$HOME/projects/FORGE"
DATA=/data/forge
CONF="$HOME/.config/forge"
UNIT_DIR="$HOME/.config/systemd/user"
PORT=8100
FUNNEL_PORT=8443
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export PATH="$HOME/.local/bin:$PATH"

step() { printf '\n==> %s\n' "$*"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

step "Code: fast-forward to the latest pushed commit"
[ -d "$APP/.git" ] || die "$APP is not a clone. Run: git clone git@github.com:saikiran-nathani/FORGE.git $APP"
git -C "$APP" pull --ff-only
git -C "$APP" log -1 --oneline

step "Python 3.12 environment (uv, no root)"
if ! command -v uv >/dev/null; then
  echo "uv not found; installing it to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
[ -x "$APP/.venv/bin/python" ] || uv venv --python 3.12 "$APP/.venv"
# Editable install: a later `git pull` + restart picks up new code. No [deep-learning]
# extra — the API trains in fast mode, so PyTorch (~2 GB) is never needed here.
uv pip install --python "$APP/.venv/bin/python" -e "$APP"
"$APP/.venv/bin/python" -c "import forge.api.app" || die "the API does not import; see the error above"

step "Data directory and settings"
mkdir -p "$DATA" "$CONF" "$UNIT_DIR"
if [ ! -f "$CONF/forge.env" ]; then
  install -m 600 "$APP/deploy/tuf/forge.env.example" "$CONF/forge.env"
  echo "Created $CONF/forge.env from the template."
fi
if grep -q "REPLACE-ME" "$CONF/forge.env"; then
  die "Set FORGE_CORS_ORIGINS in $CONF/forge.env to your Netlify URL, then re-run."
fi
grep "^FORGE_" "$CONF/forge.env"

step "Service"
install -m 644 "$APP/deploy/tuf/forge-api.service" "$UNIT_DIR/forge-api.service"
systemctl --user daemon-reload
systemctl --user enable forge-api.service >/dev/null
systemctl --user restart forge-api.service

step "Health check (http://127.0.0.1:$PORT)"
healthy=no
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:$PORT/api/v1/health"; then healthy=yes; echo; break; fi
  sleep 2
done
if [ "$healthy" != yes ]; then
  journalctl --user -u forge-api.service -n 40 --no-pager
  die "the service did not become healthy; log above"
fi
curl -fsS "http://127.0.0.1:$PORT/api/v1/experiments" | head -c 300; echo

step "Public access"
if [ "${1:-}" = "--funnel" ]; then
  # Funnel is per port: this exposes only :$FUNNEL_PORT. The hub on :443 stays tailnet-only.
  tailscale funnel --bg --https="$FUNNEL_PORT" "http://127.0.0.1:$PORT"
  tailscale funnel status
  echo
  host=$(tailscale status --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')
  echo "Public API: https://$host:$FUNNEL_PORT"
  echo "Set that as VITE_API_URL in Netlify, then trigger a new deploy (Vite bakes it in at build time)."
else
  echo "Private for now: only this machine can reach it. Re-run with --funnel to make it public."
fi

cat <<EOF

Useful commands (export XDG_RUNTIME_DIR=/run/user/\$(id -u) first in a fresh SSH shell):
  status   systemctl --user status forge-api
  logs     journalctl --user -u forge-api -f
  stop     systemctl --user stop forge-api
  private  tailscale funnel --https=$FUNNEL_PORT off
EOF
