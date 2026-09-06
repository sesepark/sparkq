# AGENTS.md — DGX Spark (이 기계)에 들어온 에이전트에게

이 기계는 SeoulLocalAgent 로봇 파이프라인의 **학습 기계**다. GPU(GB10)는 하나이고 메모리는
CPU와 통합이라, 두 작업이 겹치면 오류 없이 둘 다 느려진다. 그래서 GPU 앞에는 줄이 있다.

## 규칙 하나

> **GPU를 쓰는 것은 무엇이든 `~/sparkq`의 큐를 통해서만 띄운다.**
> 학습, 뷰어, 렌더링, 검증 스크립트, Isaac Sim을 켜는 모든 파이썬 — 예외 없음.

자세한 것(무엇이 GPU를 쓰는 것으로 치는지, 어떻게 거는지, 종류가 없을 때 어떻게 하는지,
그리고 이 규칙을 어겨서 실제로 깨졌던 일들)은 **`~/sparkq/AGENTS.md`** 에 있다. 이 기계에서
뭔가 띄우기 전에 그 파일을 읽는다.

## 자유롭게 해도 되는 것

- 파일 읽기, 로그 읽기 (`~/.sparkq/runs/<id>/run.log`, `docker exec isaac-lab-base cat …`)
- 소스 편집: `~/IsaacLab/source/…` (컨테이너 `/workspace/isaaclab/source`와 같은 디렉토리)
- `python3 ~/sparkq/sparkq.py ls|kinds|log|add|top|rm`

## 하면 안 되는 것

- `docker exec isaac-lab-base … python …` / `isaaclab.sh -p …` 를 직접 실행
- `nohup … &`, `tmux new -s train-…` 로 GPU 프로세스 만들기
- 큐가 돌리는 프로세스를 `kill` — 세우려면 `sparkq rm <id>`, 남의 잡이면 먼저 묻는다
- `/dev/nvidia*`, docker, systemd, 셸 rc 파일 등 이 기계의 설정을 바꾸기 — 먼저 묻는다

## 지금 무엇이 도는지

```bash
python3 ~/sparkq/sparkq.py ls
```

여기에 "GPU를 큐 밖의 프로세스가 쓰고 있습니다"가 보이면, 그 프로세스가 자기 것인지 먼저
확인한다. 자기 것이면 규칙을 어긴 것이다.

## 이 기계의 다른 문

팔과 카메라가 붙은 콘솔 서버는 **별개의 기계**다. 거기서 시작하는 학습도 같은 GPU를 쓰며,
`train-` 접두사의 tmux 세션으로 서로를 알아본다. 이 기계에서 `train-`으로 시작하는 세션을
직접 만들지 않는다.
