# 2026-09-06 — Isaac Sim(Kit)이 시작 단계에서 굳는다 → 원인 확정, 고침

## 증상

`isaac-rl`(학습)이나 `isaac-play`(뷰어)를 걸면 앱에는 "시작하는 중"만 계속 뜨고, `run.log`는
파이썬의 `The '--headless' CLI argument is deprecated …` 줄(7번째 줄)에서 멈춘다. 정상이면 그
바로 다음 줄에 Kit 자체의 로그(`2026-09-06T12:08:22Z [0ms] [Warning] …`)가 같은 초 안에 나온다.
GPU 사용률 0%, `nvidia-smi`에 컴퓨트 프로세스 없음. 새로 띄운 것의 약 3할이 이랬고, 같은 명령을
다시 걸면 대개 뜬다.

## 결론 먼저

**Kit 시동 중 두 스레드가 서로가 올리는 플러그인을 기다리는 교착**이다. 세마포어·libgomp·GPU와
무관하다. `omni.kit.app`의 `AppSettings`가 `std::async`로 띄운 스레드가 `carb.settings` 플러그인을
시작하는 동안(그 startup은 `carb.dictionary`를 요구한다), 주 스레드는 `carb.dictionary`를 등록하는
중이고(그 pre-startup은 settings에 기대는 인터페이스를 요구한다), 둘 다 libcarb의 같은 자리에서
"다른 스레드가 올리는 중인 플러그인"을 futex로 기다린다. 두 스레드의 시각이 겹치는 확률이 3할이다.

고침: Isaac Lab `AppLauncher`(`~/IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py`,
`_preload_core_carb_plugins`)가 `SimulationApp`을 만들기 직전, 아직 스레드가 하나뿐일 때
`carb.dictionary`·`carb.dictionary.serializer-*`·`carb.tokens`·`carb.settings`를 등록하고
`carb.settings.get_settings()`로 settings를 **시작까지** 시켜 둔다. 그러면 비동기 스레드가 기다릴
것이 없다. 학습(train.py)과 뷰어(play.py) 모두 `AppLauncher`를 지나므로 둘 다 덮는다.

## 어떻게 찾았나

### 처음 짚은 원인 둘은 틀렸다

1. SIGKILL로 죽은 Kit이 `/dev/shm/sem.carbonite-sharedmemory`를 잠근 채 남긴다(31edb1b) — 정리
   코드가 실제로 돈 뒤에도(bde85c3) 굳었다. 그 세마포어는 굳은 프로세스에서 값 1(풀림)이었다.
2. `LD_PRELOAD` libgomp의 풀 스레드에 일을 넘기고 신호를 흘렸다(이 문서의 이전 판) — 이름 없는
   스레드 20개가 0x80 간격 슬롯에서 각자 다른 주소로 자는 것은 libgomp가 아니다. libgomp 워커는
   raw futex(op 0x80)로 **같은** 배리어 주소에서 잔다. gdb로 보니 그 20개는 numpy에 딸린
   OpenBLAS의 `blas_thread_server`(op 0x189 = `pthread_cond_wait`)였고, 정상 프로세스에도 똑같이
   있는 **구경꾼**이다.

### 실제로 본 것 — gdb 스택 (세 번, 전부 같음)

굳은 프로세스 세 개(job 231550-a221 pid 263672, 231550-7b60 pid 264780·265473, 231727-946c
pid 267274)를 `/opt/gdb-host/capture.sh`(아래)로 찍었다. 파일은
`/workspace/isaaclab/logs/hang-capture/*.txt`. 매번 스레드 둘만 Carbonite식 futex
(`FUTEX_WAIT_BITSET|PRIVATE`, op 0x89, bitset 0xffffffff = `carb::thread::detail::futex_wait`)에 있었다.

```
Thread 2 (std::async 스레드):
  #0 syscall (libc)
  #1-#4 ?? (libcarb.so)                                 ← 다른 스레드가 올리는 플러그인 대기
  #5 carbOnPluginStartup (libcarb.settings.plugin.so)   ← carb.settings 시작 중 (dictionary 필요)
  #6-#8 ?? (libcarb.so)                                 ← Framework::acquireInterface → 플러그인 시작
  #9-#11 ?? (libomni.kit.app.plugin.so)
  #12-#15 std::__future_base::_Async_state_impl<…bool (omni::kit::AppSettings::*)(), AppSettings*…>::_M_run
  #16 libstdc++ thread trampoline

Thread 1 (주 스레드):
  #0 syscall (libc)
  #1-#4 ?? (libcarb.so)                                 ← 같은 주소, 같은 대기 지점
  #5-#6 ?? (libcarb.dictionary.plugin.so)               ← pluginInitialize → …ForClient → tryAcquireInterface
  #7 carbOnPluginPreStartup (libcarb.dictionary.plugin.so)
  #8-#10 ?? (libcarb.so)                                ← Framework가 dictionary를 등록하는 중
  #11-#15 ?? (libomni.kit.app.plugin.so)                ← IApp::startup
  #16-#17 omni/kit/app/_app.cpython-312 → SimulationApp._start_app (simulation_app.py:576)
  … app_launcher.py → sim_launcher.py → train.py
```

- 두 스레드의 libcarb #1-#4 주소가 **같다**. 같은 함수, 같은 대기.
- `/proc/<pid>/task/*/syscall`: 주 스레드는 자기 `[stack]` 안 주소, 스레드 2는 자기 스레드 스택
  (8MB 익명 매핑) 안 주소에서 대기 — 둘 다 스택의 지역 atomic. Carbonite의
  `carb::cpp::atomic::wait` 패턴이고, 정상 프로세스의 주 스레드도 같은 모양으로 기다리므로
  "스택 주소"만으로는 아무것도 알 수 없었다. 스택 **전체**가 필요했던 이유다.
- 열린 fd는 `carb-RStringInternals-<pid>` 셋뿐, Kit 로그 파일도 `/dev/nvidia*`도 아직이다.
  CUDA·렌더러·확장 로딩 훨씬 전, Carbonite 프레임워크의 첫 플러그인들을 올리는 단계다.
- 카운터 증거: 정상 시동에는 이 시점에 `structured log` 이름의 스레드가 있는데, 굳은 쪽에는 없다
  (structuredlog 플러그인은 settings에 기댄다).

### 왜 3할인가

`SimulationApp.__init__`은 `carb.get_framework()` 뒤 `load_plugins(["omni.kit.app.plugin"])`만 하고
`IApp::startup`을 부른다. 나머지 핵심 플러그인(dictionary, settings, tokens…)은 startup 안에서
처음 acquire 될 때 게으르게 올라간다. 그 startup이 `AppSettings` 작업을 `std::async`로 띄우는데,
그 스레드와 주 스레드가 각각 settings·dictionary를 먼저 잡으면 원이 되고, 순서가 맞으면 그냥
지나간다. 어느 쪽이 먼저인지는 스케줄링 운이다. Kit 자체의 버그이며 x86에서 안 보이는 것은
타이밍 차이일 뿐이다(Isaac Sim 6.0.1-rc.7, DGX Spark aarch64).

### 고침의 근거

플러그인이 **이미 등록되고 시작된** 뒤에는 acquire가 곧바로 돌아온다(Framework.h: "The thread
that first acquires the interface will call all load hooks … All other threads … will wait until
load hooks have been called"). 그러니 스레드가 하나뿐인 시점에 주 스레드가 두 플러그인을 다
올려 두면, 비동기 스레드는 잡을 것을 잡고 바로 돌아온다. `carb.settings.get_settings()`는
`acquire_settings_interface()`라 등록만이 아니라 시작까지 시킨다.

## 검증

같은 명령을 작게 줄인 탐침 잡(`isaac-rl num_envs=16 max_iterations=2`, 시동 21초)을 큐에 줄줄이 걸고,
시동마다 감시기가 50초 안에 Kit 로그가 없으면 gdb로 스택을 찍었다.

| | 시동 | 굳음 |
|---|---|---|
| 고치기 전 (23:15~23:26) | 13 | 4 (31%) |
| 고친 뒤 (23:26:03 배포 이후, 탐침 21 + 재시도 1 + 본 학습 1) | 23 | **0** |

굳을 확률이 그대로 31%였다면 23번 연속 정상일 확률은 0.02%다. 탐침이 남긴 로그 폴더 30개는
`logs/_trash_probes_2026-09-06/`로 옮겨 두었다(뷰어가 최신 체크포인트를 자동으로 고르므로 남겨 두면
2반복짜리를 집는다). 확인 뒤 지워도 된다.

## 지금의 대응

- 근본 고침: 위 `AppLauncher._preload_core_carb_plugins`.
- 안전망: `kinds/isaac-rl.json`, `kinds/isaac-play.json`의 시작 감시(120초 안에 Kit 로그가 없으면
  죽이고 다시, 최대 3번)는 그대로 둔다. 고침이 맞다면 이 감시는 더 이상 발동하지 않는다.
  `[sparkq] 시작 감시:` 줄이 로그에 다시 보이면 이 문서를 다시 연다.

## 도구 — 다음에 또 굳으면

컨테이너에 gdb를 넣어 두었다(`/opt/gdb-host`, 호스트 Ubuntu 24.04의 gdb 15.1과 빠진 .so 여섯 개를
복사한 것, 20MB, apt·네트워크 없이). 호스트의 `yama/ptrace_scope=1` 때문에 붙일 때는
`docker exec --privileged --user root`가 필요하다.

```bash
# 굳은 Kit 파이썬의 pid를 컨테이너 안에서 찾아 스택·매핑·스레드 상태를 한 파일에 찍는다
docker exec --privileged --user root isaac-lab-base /opt/gdb-host/capture.sh <pid> hang
# → /workspace/isaaclab/logs/hang-capture/<시각>-hang-<pid>.txt
```

gdb는 내장 파이썬을 켜야 하므로 `PYTHONHOME=/isaac-sim/kit/python`을 준다(스크립트 안에 있다).
py-spy는 파이썬 프레임이 없는 스레드를 안 보여 줘서 이 문제에는 부족했다.

## 같이 발견한 것

- `python3 sparkq.py rm <id>`가 `TimeoutError: timed out`을 찍는다. 학습 잡의 종료 트랩이
  INT → 30초 대기 → KILL로 최대 45초 걸리는데 CLI의 HTTP 대기가 그보다 짧다. 잡은 정상적으로 선다.
- 죽은 Kit이 남기는 `/dev/shm` 파일은 `carb-RStringInternals-<pid>`, `sem.carb-RStringInternals-<pid>`,
  `carb-ringbuffer-<pid>-0x…`, `sem.carbonite-sharedmemory`다. 시작 전 정리는 그대로 둔다(다른 문제의
  방어선).
- 컨테이너의 `/dev/shm`이 도커 기본값 64MB(ipc=private)다. 이번 문제와는 무관했지만 적어 둔다.
- `LD_PRELOAD` 시스템 libgomp는 torch aarch64 휠의 static TLS 문제 때문에 넣은 것이고 이번 문제와
  무관하다. 그대로 둔다.
