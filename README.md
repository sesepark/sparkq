# sparkq

DGX Spark의 GPU 하나 앞에 줄을 세우는 작은 작업 큐.

여러 학습을 손으로 하나씩 띄우고 끝나기를 기다렸다가 다음 것을 넣는 대신, 걸어 두면
순서대로 하나씩 돈다. 걸어 두고 자러 가도 된다.

표준 라이브러리만 쓴다 — 의존성이 없다. 상태는 전부 `~/.sparkq/` 아래 파일이고, 데몬은
그 파일을 읽어 tmux 안에 작업을 띄우는 얇은 층이다. `127.0.0.1`의 HTTP로만 조작한다.

이것은 [SeoulLocalAgent](https://github.com/sesepark/seoul-local-agent)의 로봇 파이프라인
가운데 **학습을 맡는 기계**에서 도는 부분이다. 팔과 카메라가 붙은 콘솔 서버는 별개다.

## 왜 만들었나

이 기계의 희소 자원은 GB10 하나다. 그래서 슬롯은 하나이고, 큐는 한 번에 하나만 돌린다.
통합메모리라 두 학습이 겹치면 OOM으로 깨지는 대신 **둘 다 스왑으로 느려지는** 방식으로
망가지는데, 그쪽이 훨씬 알아채기 어렵다. 겹치지 않게 하는 것이 이 프로그램의 전부다.

## 설계

- **슬롯 하나, 비선점.** 시작한 작업은 끝나거나 사람이 세울 때까지 둔다. 학습은 몇 시간
  짜리라 중간에 뺏으면 처음부터 다시 해야 한다. 아직 시작하지 않은 줄은 얼마든지 다시
  세울 수 있다(`top`).
- **상태는 디스크에.** `~/.sparkq/queue/`의 파일 이름이 곧 순서(우선순위 → 들어온 시각).
  실행 중인 것은 `~/.sparkq/runs/<id>/`에 `job.json`·`run.log`·`progress.json`으로 남는다.
  데몬이 `systemctl --user restart` 돼도 도는 학습은 tmux 안에서 그대로 돌고, 올라온
  데몬이 세션 이름으로 다시 찾아 붙는다.
- **GPU가 실제로 비었을 때만 다음을 꺼낸다.** 앞 작업이 끝났는지가 아니라 `nvidia-smi`에
  컴퓨트 프로세스가 하나도 없는지를 본다. 사람이 터미널에서 직접 띄운 학습 위에 올라타지
  않기 위해서다. nvidia-smi를 못 읽으면(빈 목록과 다르다) 시작하지 않고 기다린다.
- **작업 종류는 파일로 늘린다.** `kinds/*.json` 하나가 종류 하나다. 새 실험을 걸 수 있게
  하는 데 코드를 고칠 필요가 없다. 대신 그 JSON이 셸 명령을 만들므로, 값은 목록(`enum`)과
  범위(`int`)와 이름(`name`)으로만 받는다 — 자유 문자열 형식은 일부러 없다. 그것이 들어가는
  순간 이 큐는 원격 셸이 된다.

## 설치

```bash
bash install.sh
```

`loginctl enable-linger`를 켜고(한 번 sudo) 사용자 systemd 서비스로 올린다. linger가 없으면
로그아웃과 동시에 사용자 유닛이 죽어, 밤새 도는 것이 목적인 이 큐가 의미를 잃는다.

되돌리기:

```bash
systemctl --user disable --now sparkq
sudo loginctl disable-linger "$USER"
```

## 쓰기

```bash
sparkq ls                 # 지금 도는 것과 대기열, 최근 끝난 것
sparkq kinds              # 걸 수 있는 작업 종류
sparkq add <종류> 이름=값 …  # 줄 맨 뒤에 세운다
sparkq top <id>           # 아직 시작 안 한 작업을 맨 앞으로
sparkq rm <id>            # 대기 취소 또는 도는 작업 중지
sparkq log <id>           # 로그 꼬리
sparkq pause / resume     # 다음 작업을 꺼낼지 말지
```

기본으로 딸려 오는 두 종류:

```bash
sparkq add lerobot-train dataset=<이름> policy=act        # LeRobot 정책 학습 (act | smolvla)
sparkq add isaac-rl task=Isaac-Lift-Cube-SO101-v0 num_envs=4096 max_iterations=2500
```

## HTTP API

`127.0.0.1:8092`(기본). LAN에 여는 길은 만들지 않는다 — 신뢰 경계는 이 앞의 SSH 터널이다.

| 메서드 | 경로 | 하는 일 |
|---|---|---|
| GET | `/api/status` | 기계 한 줌: GPU 온도·전력·사용률, CPU·메모리 사용률, 디스크, 대기 개수 |
| GET | `/api/kinds` | 걸 수 있는 작업 종류와 칸 명세 |
| GET | `/api/datasets` | `~/data/soarm` 아래에 와 있는 데이터셋 |
| GET | `/api/queue` | 도는 것 1 + 대기열 + 최근 끝난 것 + 큐 밖의 GPU 프로세스 |
| POST | `/api/queue` | `{"kind": …, "params": {…}}` |
| DELETE | `/api/queue/{id}` | 대기면 빼고, 도는 중이면 세운다 |
| POST | `/api/queue/{id}/top` | 맨 앞으로 |
| POST | `/api/queue/pause` | `{"paused": true\|false}` |
| GET | `/api/queue/{id}/log` | 로그 꼬리 |

## 작업 종류 만들기

`kinds/`에 JSON 하나를 놓으면 된다. `${이름}`은 `string.Template`으로 치환되고(모르는 이름은
건드리지 않는다), 만들어 낸 셸 명령은 **걸 때** 확정되어 작업 파일에 그대로 적힌다 — 자는
동안 도는 것들이므로, 무엇이 돌 것인지 미리 읽을 수 있어야 한다.

```json
{
  "kind": "lerobot-train",
  "title": "${policy} 학습 · ${dataset}",
  "label": "데이터셋 학습",
  "fields": [
    {"name": "dataset", "label": "데이터셋", "type": "name", "source": "datasets"},
    {"name": "policy", "label": "정책", "type": "enum", "values": ["act", "smolvla"],
     "default": "act", "presets": {"act": {"steps": 100000, "batch_size": 64}}}
  ],
  "derived": [{"name": "run", "template": "${dataset_short}__${policy}__${token}"}],
  "session": "train-${run}",
  "progress": "lerobot",
  "run": "…셸 스크립트…"
}
```

세션 이름은 반드시 `train-`으로 시작해야 한다(데몬이 강제한다). 같은 GPU를 쓰는 다른 문 —
콘솔 서버의 학습 시작 — 이 그 접두사로만 "이미 도는 학습"을 알아보기 때문이다.
