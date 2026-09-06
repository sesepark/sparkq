# 2026-09-06 — Isaac Sim(Kit)이 시작 단계에서 굳는다

## 증상

`isaac-rl`(학습)이나 `isaac-play`(뷰어)를 걸면 앱에는 "시작하는 중"만 계속 뜨고, `run.log`는
파이썬의 `The '--headless' CLI argument is deprecated …` 줄(7번째 줄)에서 멈춘다. 정상이면 그
바로 다음 줄에 Kit 자체의 로그(`2026-09-06T12:08:22Z [0ms] [Warning] …`)가 같은 초 안에 나온다.
GPU 사용률 0%, `nvidia-smi`에 컴퓨트 프로세스 없음. 이날 새로 띄운 것의 약 3할(다섯 번)이
이랬고, 같은 명령을 다시 걸면 대개 뜬다.

## 처음 짚은 원인은 틀렸다

처음에는 SIGKILL로 죽은 Kit이 `/dev/shm/sem.carbonite-sharedmemory`(Carbonite 공유메모리의 전역
세마포어)를 잠근 채 남기고, 다음 Kit이 그것을 기다린다고 봤다(31edb1b). 시작 전 정리에 그 파일을
지우는 코드를 넣었는데, 그 코드는 `pgrep -f kit/python/bin/python3`이 **자기 자신(bash -c의
인자에 패턴이 들어 있다)** 을 잡아 늘 건너뛰고 있었다(bde85c3에서 고침). 고친 뒤 정리가 실제로
돌고("전역 세마포어까지 제거"가 로그에 찍힘) 새 Kit이 세마포어를 새로 만들었는데도 **또 굳었다.**

## 실제로 본 것 (job 20260906_211058-d19e, 컨테이너 PID 241730)

py-spy를 컨테이너에 넣고(`docker cp`) `docker exec --privileged --user root`로 붙였다.
호스트에서는 `yama/ptrace_scope=1`이라 같은 uid여도 gdb가 못 붙는다.

```
Thread 241730 "MainThread"
    syscall (libc.so.6)                        ← futex
    (libcarb.so) ×4
    (libcarb.dictionary.plugin.so) ×2
    carbOnPluginPreStartup (libcarb.dictionary.plugin.so)
    (libcarb.so) ×3
    (libomni.kit.app.plugin.so) ×5
    _start_app (simulation_app/simulation_app.py:576)
    __init__ (isaaclab/app/app_launcher.py:300)
    launch_simulation (isaaclab_tasks/utils/sim_launcher.py:466)
    main (train.py:127)
```

`/proc/241730/task/241730/syscall`:

```
98 0xfffff796e144 0x89 0x0 0x0 0x0 0xffffffff …
   uaddr = 주 스레드 자기 스택 안        op 0x89 = FUTEX_WAIT_BITSET | FUTEX_PRIVATE_FLAG
   timeout = NULL(무한)                  bitset = 0xffffffff
```

- 기다리는 주소가 **자기 스택**이고 `PRIVATE` 플래그라 **다른 프로세스는 깨울 수 없다.**
  즉 공유메모리·세마포어·GPU 드라이버가 아니라 **프로세스 안의 교착**이다.
- 나머지 스레드 20개는 전부 이름 없는(`comm=python3`) 스레드로, 익명 메모리의 0x80 간격 슬롯에
  각각 futex 대기 중 — 코어 수(20)와 같은 유휴 스레드 풀이다. `LD_PRELOAD`로 넣는 libgomp의
  풀로 보인다(torch가 Kit보다 먼저 import 된다; `libtorch`, `libnvshmem`이 이미 매핑돼 있었다).
- 열린 fd는 `carb-RStringInternals-241730` 셋뿐이다. Kit 로그 파일도, `/dev/nvidia*`도 아직
  안 열었다 — CUDA 초기화 훨씬 전, Carbonite 프레임워크의 첫 플러그인(carb.dictionary)을 올리는
  단계다.
- 이 프로세스가 만든 세마포어들은 잠겨 있지 않았다. `sem_open`은 임시 이름으로 만들어 `link`
  하므로 맵에는 `sem.kR9aaa (deleted)`로 보이는데, inode(803)가 `/dev/shm/sem.carbonite-sharedmemory`와
  같고 값은 1(풀림), RStringInternals 쪽(804)은 2였다.

## 모르는 것

carb.dictionary의 pre-startup이 무엇을 기다리는지. 신호를 줄 스레드가 없는데 기다리므로,
스레드를 만들다 실패했거나(반환값을 안 보는 코드), 이미 잠든 풀 스레드에 일을 넘기고 깨우는
신호를 흘렸거나(lost wakeup) 둘 중 하나로 보인다. 열에 일곱은 뜨므로 경쟁 조건이다. 심볼이
없어 libcarb 안의 함수는 못 읽었다. 다음에 또 굳으면 아래를 더 본다.

- `LD_PRELOAD=libgomp` 없이 띄우면 재현되는지(그 preload는 aarch64 torch 때문에 넣은 것이다).
- `OMP_NUM_THREADS=1`로 풀 스레드를 없애면 재현되는지.
- 컨테이너에 gdb를 넣고(`apt install gdb`, 사용자에게 먼저 묻는다) 풀 스레드 20개의 스택을
  읽는다. py-spy는 파이썬 프레임이 없는 스레드를 보여 주지 않는다.

## 지금의 대응 (kinds/isaac-rl.json, kinds/isaac-play.json)

시작 감시. 실행을 백그라운드로 띄우고 `run.log`에 Kit 로그 줄(ISO 시각으로 시작하는 줄)이
120초 안에 나오는지 본다. 안 나오면 `pkill -KILL` 하고, 프로세스가 다 사라지길 기다린 뒤,
`/dev/shm`을 치우고 다시 띄운다. 최대 3번. 무엇을 했는지는 `[sparkq] 시작 감시: …`로 로그에
남긴다. 3할이 굳는다면 세 번 연속 굳을 확률은 3% 정도다.

## 같이 발견한 것

- `python3 sparkq.py rm <id>`가 `TimeoutError: timed out`을 찍는다. 학습 잡의 종료 트랩이
  INT → 30초 대기 → KILL로 최대 45초 걸리는데 CLI의 HTTP 대기가 그보다 짧다. 잡은 정상적으로
  선다. 앱에서도 같은 시간 동안 기다리게 된다.
- 죽은 Kit이 남기는 `/dev/shm` 파일은 `carb-RStringInternals-<pid>`, `sem.carb-RStringInternals-<pid>`,
  `carb-ringbuffer-<pid>-0x…`, `sem.carbonite-sharedmemory`다. 시작 전 정리는 이제 실제로 돈다.
