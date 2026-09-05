#!/usr/bin/env bash
# sparkq를 사용자 systemd 서비스로 올린다. 되돌리려면 아래 두 줄이면 된다.
#   systemctl --user disable --now sparkq
#   sudo loginctl disable-linger "$USER"
set -euo pipefail

mkdir -p ~/.sparkq/queue ~/.sparkq/runs ~/.config/systemd/user
install -m 644 "$(dirname "$0")/sparkq.service" ~/.config/systemd/user/sparkq.service
ln -sf "$(cd "$(dirname "$0")" && pwd)/sparkq.py" ~/.local/bin/sparkq 2>/dev/null || {
  mkdir -p ~/.local/bin
  ln -sf "$(cd "$(dirname "$0")" && pwd)/sparkq.py" ~/.local/bin/sparkq
}

# linger가 없으면 로그아웃과 동시에 사용자 유닛이 전부 죽는다. 밤새 도는 것이 목적이므로
# 이것이 켜져 있지 않으면 이 큐는 의미가 없다.
if [ "$(loginctl show-user "$USER" -p Linger --value)" != "yes" ]; then
  echo "== linger를 켭니다 (sudo가 필요합니다) =="
  sudo loginctl enable-linger "$USER"
fi

systemctl --user daemon-reload
systemctl --user enable --now sparkq
sleep 1
systemctl --user --no-pager status sparkq | head -12
