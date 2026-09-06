#!/usr/bin/env bash
# sparkq를 맥의 LaunchAgent로 올린다. 되돌리려면 아래 두 줄이면 된다.
#   launchctl bootout gui/$(id -u)/com.sesepark.sparkq
#   rm ~/Library/LaunchAgents/com.sesepark.sparkq.plist
set -euo pipefail

LABEL=com.sesepark.sparkq
HERE="$(cd "$(dirname "$0")" && pwd)"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

# 데몬 자신은 표준 라이브러리만 쓴다. homebrew 쪽이 있으면 그것을 쓰고, 없으면 시스템
# 파이썬으로 간다 — 3.9에서도 도는 것을 확인해 두었다.
PYTHON=/usr/bin/python3
for candidate in /opt/homebrew/bin/python3 /usr/local/bin/python3; do
  [ -x "$candidate" ] && PYTHON="$candidate" && break
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "== tmux가 없습니다. 학습은 tmux 안에서 돌므로 먼저 깔아야 합니다 =="
  echo "   brew install tmux"
  exit 1
fi

mkdir -p ~/.sparkq/queue ~/.sparkq/runs ~/Library/LaunchAgents ~/data/soarm ~/outputs

sed -e "s|__HOME__|$HOME|g" -e "s|__ROOT__|$HERE|g" -e "s|__PYTHON__|$PYTHON|g" \
  "$HERE/sparkq.macos.plist" > "$PLIST"

mkdir -p ~/.local/bin
ln -sf "$HERE/sparkq.py" ~/.local/bin/sparkq

# 이미 올라와 있으면 내리고 다시 올린다. bootstrap은 두 번 부르면 실패한다.
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl kickstart "gui/$(id -u)/$LABEL" >/dev/null

sleep 1
if curl -fsS --max-time 5 http://127.0.0.1:8093/api/status >/dev/null 2>&1; then
  echo "== 올라왔습니다. 127.0.0.1:8093 =="
  echo "   sparkq는 SPARKQ_PORT=8093을 붙여 부릅니다:"
  echo "     SPARKQ_PORT=8093 ~/.local/bin/sparkq ls"
else
  echo "== 아직 답하지 않습니다. 로그를 보세요: ~/.sparkq/daemon.log =="
  tail -20 ~/.sparkq/daemon.log 2>/dev/null || true
  exit 1
fi
