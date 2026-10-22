# sparkq 설치와 사용법

[프로젝트 소개](../README.md) · [에이전트 규칙](../AGENTS.md)

## 설치

### 리눅스 (systemd)

```bash
bash install.sh
```

`loginctl enable-linger`를 켜고(한 번 sudo) 사용자 systemd 서비스로 올림. linger가 없으면
로그아웃과 동시에 사용자 유닛이 죽어, 사람이 보지 않는 동안 도는 것이 목적인 이 큐가 의미를 잃음.

되돌리기:

```bash
systemctl --user disable --now sparkq
sudo loginctl disable-linger "$USER"
```

### 맥 (launchd)

```bash
brew install tmux          # 학습은 tmux 안에서 돈다
bash install-macos.sh
```

포트는 **8093**임. 8092를 쓰지 않는 이유는 맥에서 Spark의 큐로 가는 SSH 터널이 그 번호를
이미 잡고 있어서, 같은 번호면 둘 중 하나가 뜨지 못하기 때문임.

`linger`에 해당하는 것이 없음 — LaunchAgent는 **로그인해 있는 동안** 돔. 로그인 화면에
머물러 있으면 큐도 서지 않으므로 밤새 돌릴 때는 로그인한 채로 두어야 함. 그리고 외장
디스플레이 없이 **뚜껑을 닫으면 `caffeinate`로도 못 막음** — 뚜껑은 열어 두어야 함.

되돌리기:

```bash
launchctl bootout gui/$(id -u)/com.sesepark.sparkq
rm ~/Library/LaunchAgents/com.sesepark.sparkq.plist
```

## 쓰기

```bash
sparkq ls                 # 지금 도는 것과 대기열, 최근 끝난 것
sparkq kinds              # 걸 수 있는 작업 종류
sparkq add <종류> 이름=값 …  # 줄 맨 뒤에 세운다
sparkq top <id>           # 아직 시작 안 한 작업을 맨 앞으로
sparkq mv <id> [<앞>]     # 그 작업 바로 앞으로 (뒤 인자가 없으면 맨 뒤로)
sparkq rm <id>            # 대기 취소 또는 도는 작업 중지
sparkq rename <id> <이름>   # 끝난 작업의 이름 바꾸기 (기록의 제목만)
sparkq forget <id>        # 끝난 작업을 내역에서 빼기 (기록은 ~/.sparkq/trash 로)
sparkq log <id>           # 로그 꼬리
sparkq pause / resume     # 다음 작업을 꺼낼지 말지
```

맥에서는 포트가 다르므로 앞에 붙여 부름:

```bash
SPARKQ_PORT=8093 sparkq ls
```

기본으로 딸려 오는 두 종류:

```bash
sparkq add lerobot-train dataset=<이름> policy=act        # LeRobot 정책 학습 (act | smolvla)
sparkq add isaac-rl task=Isaac-Lift-Cube-SO101-v0 num_envs=4096 max_iterations=2500
```

![맥 앱이 이 큐를 보여 주는 화면](../docs/images/sparkq-in-app.png)

<sub>큐를 읽는 쪽의 예 — [seoul-local-agent](https://github.com/sesepark/seoul-local-agent)의
`학습 서버` 화면이 `/api/queue`와 `/api/status`를 그대로 그린 것. 진행·남은 시간은 `progress.json`,
GPU·온도·전력은 `/api/status`에서 옴.</sub>

## HTTP API

`127.0.0.1:8092`(기본). LAN에 여는 길은 만들지 않음 — 신뢰 경계는 이 앞의 SSH 터널이다.

| 메서드 | 경로 | 하는 일 |
|---|---|---|
| GET | `/api/status` | 기계 한 줌: GPU 온도·전력·사용률, CPU·메모리 사용률, 디스크, 대기 개수 |
| GET | `/api/kinds` | 걸 수 있는 작업 종류와 칸 명세 |
| GET | `/api/datasets` | `~/data/soarm` 아래에 와 있는 데이터셋 |
| GET | `/api/runs` | 학습이 남긴 것 — 실행마다 체크포인트·크기·지금 쓰는 중인지 |
| DELETE | `/api/runs/{run}` | 실행 하나를 통째로 (옆자리 `.runs/{run}`의 로그도 함께) |
| DELETE | `/api/runs/{run}/checkpoints/{step}` | 체크포인트 하나 |
| DELETE | `/api/runs/{run}/checkpoints/{step}/training_state` | optimizer 상태만 — 가중치는 남는다 |
| GET | `/api/queue` | 도는 것 1 + 대기열 + 최근 끝난 것 + 큐 밖의 GPU 프로세스 |
| POST | `/api/queue` | `{"kind": …, "params": {…}}` |
| DELETE | `/api/queue/{id}` | 대기면 빼고, 도는 중이면 세운다 |
| POST | `/api/queue/{id}/top` | 맨 앞으로 |
| POST | `/api/queue/{id}/move` | `{"before": "<id>"}` — 그 작업 **바로 앞**으로. `{"before": null}`이면 맨 뒤로. 자리를 번호가 아니라 다른 작업의 이름으로 받는다 — 옮기려는 사이에 앞의 것이 시작하면 번호는 다른 자리를 가리키지만 "저것 앞"은 그대로다 |
| POST | `/api/queue/pause` | `{"paused": true\|false}` |
| GET | `/api/queue/{id}/log` | 로그 꼬리 |
| GET | `/api/queue/{id}/series` | 값의 흐름 — LeRobot은 손실·검증 손실, Isaac은 평균 보상 |

### 학습이 남긴 것

이 기계의 GPU를 큐가 소유하듯, 이 기계의 `~/outputs`도 큐가 소유한다. 다른 기계가 ssh로
들어와 지우는 구조를 만들지 않는 이유는 같다 — 무엇이 지금 쓰이고 있는지 아는 곳이 여기뿐이다.

`/api/runs`의 한 줄은 실행 하나이고, 체크포인트마다 **가중치와 optimizer 상태를 따로** 센다.
또한 데이터셋 규모, 기반 모델, 학습 범위, 배치·optimizer·검증 설정과 손실 곡선을 함께
돌려준다. 각 체크포인트의 `loss`·`eval_loss`는 그 스텝 또는 그 직전 마지막 실측값이다.

```json
{"name": "soarm101_…__smolvla__e315", "policy": "smolvla", "step": 20000, "steps": 20000,
 "bytes": 5033164800, "in_use": null,
 "checkpoints": [{"step": "005000", "model_bytes": 907018240, "state_bytes": 413138944,
                  "bytes": 1320157184, "finished_at": 1757…}]}
```

두 숫자를 나눠 두는 이유는 잃는 것이 다르기 때문이다. `training_state`는 **이어붙일 때만**
쓰이고 추론에는 필요 없는데 체크포인트의 3분의 1쯤을 차지한다(실측 865MB 대 394MB). 그것만
지우면 이어붙일 권리를 버리고 가중치는 남으므로, 그 체크포인트는 여전히 팔로 보내 돌릴 수 있다.

`in_use`가 비어 있지 않으면 지우기는 409로 거절된다. 도는 학습뿐 아니라 **아직 시작하지 않은**
작업이 가리키는 실행도 막는다 — 대기 중인 `lerobot-resume`이 가리키는 폴더를 지우면 그 작업은
새벽에 시작해 몇 초 만에 죽고, 아침에 남는 것은 실패 한 줄과 날아간 밤 하나다.

체크포인트 하나를 지우면 `checkpoints/last` 링크를 남은 것 가운데 마지막으로 옮긴다. 그러지
않으면 끊어진 링크가 남고, 실행을 알아보는 표지가 그 링크 너머의 `train_config.json`이라
**실행 전체가 목록에서 사라진다** — 지운 것은 체크포인트 하나였는데.

```bash
sparkq runs                              # 무엇이 얼마를 차지하고 있나
sparkq rm-ckpt <run> <step> --state-only # 이어붙이기만 버리고 가중치는 남긴다
sparkq rm-run  <run>                     # 통째로
```

## 작업 종류 만들기

`kinds/`에 JSON 하나를 놓으면 됨.

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

세션 이름은 반드시 `train-`으로 시작해야 함(데몬이 강제한다). 같은 GPU를 쓰는 다른 문 — 콘솔
서버의 학습 시작 — 이 그 접두사로만 "이미 도는 학습"을 알아보기 때문임.

### 스스로 정리하는 작업이면 `stop_grace_seconds`

중지는 세션에 `C-c`를 주는 것으로 시작한다. 작업이 EXIT trap으로 자기 뒤를 치운다면, 큐는 그
정리가 끝날 때까지 기다렸다가 세션이 그래도 남아 있을 때만 죽인다. 그 기다림의 길이가
`stop_grace_seconds`(기본 10초, 상한 60초)임.

이 값이 정리 스크립트보다 짧으면 정리가 **중간에 끊긴다.** 실제로 그랬다 — Isaac 뷰어의 trap은
컨테이너 안에 `pkill -INT`를 보내고 5초를 기다렸다가 `pkill -KILL`을 보내는데, 큐가 2초 만에
세션을 죽여 그 KILL이 영영 실행되지 않았다. 남은 `play.py`가 GPU를 쥐면 큐는 다음 작업을
꺼내지 못한다. `docker exec`는 클라이언트가 죽어도 컨테이너 안으로 신호를 보내지 않으므로,
컨테이너 안을 치울 수 있는 것은 작업 자신의 trap뿐이고 큐가 할 수 있는 유일한 도움은
**기다려 주는 것**임.

빨리 끝나는 작업이 손해 보지는 않음. 세션이 사라지는 즉시 돌아온다.

### 딸린 곁다리는 학습과 함께 선다

종류 파일이 `watches`로 말한 종류(뷰어면 `["isaac-rl"]`)의 학습이 도는 중에 띄운 곁다리는
`job.json`의 `follows`에 그 학습 번호를 적는다. 그 학습이 서거나 끝나면(중지·완료·실패 모두)
큐가 그 곁다리도 바로 세운다(`stop_following_side`). 학습은 섰는데 뷰어만 남아 GPU를 쥔 채
앱의 "바로보기"에 살아 있는 모양을 없애려는 것이다(2026-09-07). 학습이 끝난 **뒤에** 띄운
뷰어는 `follows`가 없으므로 시한까지 그대로 돈다.

### 흔들림을 못 견디는 곁다리면 `preempt`

```json
{"kind": "soarm-policy", "lane": "side", "limit_seconds": 600, "preempt": true}
```

`preempt: true`인 곁다리는 뜨기 직전에 **도는 학습을 얼리고**(`SIGSTOP`), 끝나면 깨운다
(`SIGCONT`). 죽이는 것이 아니라 멈추는 것이라 스텝도 옵티마이저 상태도 잃지 않음. 통합메모리라
얼어붙은 학습이 쥔 자리는 그대로 두고 GPU 연산만 비켜 주는 것이 가능해서 성립함.

큰 정책 추론처럼 **메모리까지 비워야 하는** 곁다리는 `"preempt": "checkpoint"`를 쓴다.
큐의 LeRobot launcher가 `SIGUSR1`을 받으면 현재 optimizer update를 끝낸 뒤 모델과 optimizer
상태를 정상 체크포인트로 저장하고 종료한다. 곁다리가 끝나면 같은 실행·같은 목표 스텝을
`--resume=true`로 다시 띄운다. FastWAM 학습을 SIGSTOP만 해 둔 채 FastWAM 추론을 함께 올렸을
때 통합메모리가 121GiB까지 차고 OOM killer가 추론 서버를 죽인 사건(2026-09-20) 때문에
`soarm-policy`는 이 모드를 쓴다.

`lane: "side"`에서만 쓸 수 있음. 큐 종류가 이것을 주장하면 종류 파일 자체를 읽지 않는다 —
멈춰 두기는 **시한이 있는** 레인이라야 성립하고, 깨워 줄 시한이 없으면 얼어붙은 학습이
그대로 남기 때문임.

무엇에 켜야 하는가: **사람이 보는 것**(뷰어)은 켜지 않는다. 겹쳐 돌아 조금 느려져도 화면이
달라지지 않고, 지금처럼 "느려짐 %"를 재서 보여 주는 쪽이 낫다. **재는 것**(정책 추론)은
켠다. 옆에서 학습이 돌면 추론 지연이 흔들리고, 그 흔들림은 곧 시행 결과의 오염이라서 그
시간에 잰 성공률을 못 믿게 됨.

자세한 것(왜 학습 종류로 나누지 않았는지, 깨우는 손을 왜 셋 두었는지)은
[docs/2026-09-07-preempt.md](../docs/2026-09-07-preempt.md).

## 한계와 주의

- **이 기계 한 대를 위해 만든 것임.** 슬롯 하나·GPU 하나를 전제로 하고, 여러 노드나 여러 GPU를
  나눠 쓰는 스케줄링은 없음.
- 인증이 없음. `127.0.0.1`에만 붙고, 밖에서 오는 것은 SSH 터널을 지나야 함. LAN에 여는 설정은
  일부러 만들지 않았음.
- 실패한 작업을 자동으로 다시 걸지 않음. 왜 죽었는지 사람이 읽고 다시 거는 편이 낫다고 봄.
- 딸려 오는 두 작업 종류는 이 기계의 환경을 가정함. `kinds/lerobot-train.json`이
  `~/venvs/lerobot`을 활성화하고, 데이터셋은 `~/data/soarm` 아래에서 찾음(`SPARKQ_DATASETS`로
  바꿀 수 있음). 다른 기계에서 쓰려면 그 JSON을 먼저 고쳐야 함.
