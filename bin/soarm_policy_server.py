#!/usr/bin/env python3
"""LeRobot의 policy_server를 두 가지 점에서 감싼다 — 적재 캐시와 RTC.

## 1. 같은 체크포인트를 두 번 적재하지 않는다

`SendPolicyInstructions`는 클라이언트가 붙을 때마다 `from_pretrained()`를 무조건 다시
부른다(`policy_server.py:152`). π₀.₅ 한 벌(9.35GB)에 이 기계에서 **131초**가 걸리고, 시행마다
클라이언트가 새로 붙으므로 자리 5곳 × 4회짜리 성공률 측정이면 적재에만 44분이 간다.

## 2. 새 청크를 앞 계획에 이어 붙인다 (Real-Time Chunking)

`RobotClientConfig`에는 RTC 자리가 없다. 그래서 원격 추론은 **지금 팔이 무엇을 실행 중인지
모른 채** 청크를 만들고, 클라이언트가 겹치는 구간을 `0.3*old + 0.7*new`로 섞어 갈아 끼운다.
2026-09-07 실측으로 그 교체가 **24프레임(0.80초)마다** 일어났고 팔이 그 주기로 툭툭 끊겼다.
2026-09-06에 로컬 경로에서 겪은 것과 같은 고장이다(그때는 0.73초). 로컬은 그때
`RTC_EXECUTION_HORIZON = 25`로 고쳤는데, 원격은 그 수정을 받은 적이 없다.

RTC에 필요한 두 값은 **서버가 스스로 구할 수 있다.** 클라이언트가 관측마다 `timestep`을
싣고 오므로, 서버는 자기가 마지막으로 낸 청크가 어느 timestep에서 시작했는지와 견주어
"아직 실행되지 않은 꼬리"를 그대로 잘라낼 수 있다. 프로토콜을 바꿀 필요가 없다.

체크포인트에는 `rtc_config`가 없어서(`PI05Config.rtc_config`의 기본값은 `None`) RTC 처리기
자체가 만들어지지 않는다. 여기서 만들어 넣는다. 기본 `execution_horizon`은 **10**인데, 그
값이 2026-09-06 사고의 원인이었으므로 쓰지 않는다.

2026-09-09에 이 다리가 GR00T에서 한 번도 걸리지 않았다는 것이 드러났다. `_enable_rtc`가
π₀ 계열의 `init_rtc_processor()`만 알고 있었고, GR00T에는 그 메서드가 없어 `AttributeError`가
났으며 그것을 "RTC를 모르는 정책"으로 읽어 조용히 껐다. 실제로는 GR00T도 RTC를 받는다 —
이름표가 `supports_rtc()`로 다를 뿐이다. 그래서 롤아웃 하나의 청크 155개가 전부 앞 계획 없이
새로 만들어졌고, 계획이 초당 2.8회 통째로 갈리면서 팔이 그 주기로 끊겼다. 실측: 제어 틱의
21.6%에서 안전 클램프가 걸렸고 그 76%가 청크 교체 뒤 6틱 안에 몰려 있었다. 조용한 폴백은
이렇게 오래 숨는다 — 그래서 지금은 켜진 방식까지 로그에 적는다.

2026-09-09에 한 가지가 더 드러났다. RTC를 켠 뒤에도 팔이 7틱(238ms)마다 튀었는데,
서버가 RTC에 넘기는 `inference_delay`가 **추론 시간만** 세고 있었기 때문이다(5프레임).
클라이언트가 청크를 기다리는 동안 실제로 실행하는 액션은 중앙값 7·90퍼센타일 13프레임이라,
실행이 시작되는 자리가 고정 구간 밖이었다. 이제 클라이언트가 실제로 소비한 프레임 수를
관측값으로 써서 그 자리를 덮는다.

고치는 자리를 site-packages가 아니라 여기로 잡은 이유: 그 파일을 고치면 `uv sync`나
lerobot 재설치가 조용히 되돌린다.
"""

from __future__ import annotations

import math
import time

from lerobot.async_inference import policy_server

#: 새 청크를 앞 계획에 붙들어 두는 구간(프레임).
#:
#: 로컬 경로가 2026-09-06에 고른 것과 같은 값이고, 이유도 같다. 큐가 버리는 앞부분(추론이
#: 걸린 프레임 수)보다 커야 실제로 실행되는 구간에 연속성 제약이 남는다. 이 기계의 원격
#: 왕복은 0.33~0.67초, 30fps로 10~20프레임이므로 25가 그 위에 선다. lerobot은 남은 꼬리가
#: 이보다 짧으면 스스로 꼬리 길이로 줄인다(`modeling_rtc.py`), 그래서 위쪽만 정하면 된다.
RTC_EXECUTION_HORIZON = 25

_original_get_policy_class = policy_server.get_policy_class
_original_get_action_chunk = policy_server.PolicyServer._get_action_chunk
_original_predict_action_chunk = policy_server.PolicyServer._predict_action_chunk
_original_reset_server = policy_server.PolicyServer._reset_server

_LOADED: dict[tuple[str, str], object] = {}


class _CachedPolicyClass:
    """`get_policy_class`가 돌려주던 클래스의 자리에 서서, 같은 경로면 든 것을 다시 준다."""

    def __init__(self, policy_class: type) -> None:
        self._policy_class = policy_class

    def from_pretrained(self, path, *args, **kwargs):
        key = (self._policy_class.__name__, str(path))
        cached = _LOADED.get(key)
        if cached is not None:
            print(f"[soarm] 이미 든 정책을 그대로 씁니다 — 적재 건너뜀: {key[1]}", flush=True)
            return cached
        _LOADED.clear()
        print(f"[soarm] 정책을 처음 듭니다 — 이 체크포인트는 이번 한 번만 오래 걸립니다: {key[1]}", flush=True)
        policy = self._policy_class.from_pretrained(path, *args, **kwargs)
        _enable_rtc(policy)
        _LOADED[key] = policy
        return policy

    def __getattr__(self, name: str):
        return getattr(self._policy_class, name)


def _rtc_mode(policy) -> str:
    """이 정책이 RTC를 어떤 방식으로 받는지. `_enable_rtc`가 정해서 붙여 둔다."""
    return getattr(policy, "_soarm_rtc_mode", "none")


def _enable_rtc(policy) -> None:
    """RTC를 켠다. 정책 계열마다 받는 방식이 둘로 갈린다.

    π₀ 계열은 `init_rtc_processor()`가 만들어 주는 `rtc_processor` 객체가 유도를
    맡는다. GR00T N1.7에는 그런 객체가 없다 — `supports_rtc()`가 참이고,
    `predict_action_chunk`가 `inference_delay`·`prev_chunk_left_over`를 직접 받아
    네이티브 겹침 옵션으로 바꾼다(`policies/groot/modeling_groot.py`의
    `_prepare_n1_7_rtc_inputs`). 부르는 쪽 시그니처는 두 계열이 같으므로,
    갈라지는 것은 **켜졌는지 판정하는 방법**뿐이다.

    `config.rtc_config`는 두 계열 모두가 읽으므로 언제나 넣는다. GR00T는 거기서
    `execution_horizon`을 꺼내 `rtc_overlap_steps`로 쓰고, 없으면 꼬리 전체를
    제약으로 삼는다 — 그러면 새 관측이 1초 넘게 무시된다.
    """
    try:
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        policy.config.rtc_config = RTCConfig(execution_horizon=RTC_EXECUTION_HORIZON)
    except Exception as error:  # RTC를 모르는 lerobot 버전
        policy._soarm_rtc_mode = "none"
        print(f"[soarm] RTC 설정을 넣지 못했습니다({error}). 이어 붙이기 없이 돕니다.", flush=True)
        return

    if hasattr(policy, "init_rtc_processor"):
        try:
            policy.init_rtc_processor()
        except Exception as error:
            print(f"[soarm] RTC 처리기를 만들지 못했습니다({error}).", flush=True)
        else:
            if getattr(policy, "rtc_processor", None) is not None:
                policy._soarm_rtc_mode = "processor"
                print(
                    f"[soarm] RTC 켜짐(처리기) · execution_horizon={RTC_EXECUTION_HORIZON}",
                    flush=True,
                )
                return

    supports = getattr(policy, "supports_rtc", None)
    try:
        native = bool(supports()) if callable(supports) else bool(supports)
    except Exception as error:
        print(f"[soarm] RTC 지원 여부를 묻지 못했습니다({error}).", flush=True)
        native = False

    if native:
        policy._soarm_rtc_mode = "native"
        print(
            f"[soarm] RTC 켜짐(정책 자체 경로) · execution_horizon={RTC_EXECUTION_HORIZON}",
            flush=True,
        )
        return

    policy._soarm_rtc_mode = "none"
    print("[soarm] 이 정책은 RTC를 받지 않습니다. 이어 붙이기 없이 돕니다.", flush=True)


class _RealTimeChunking:
    """마지막으로 낸 청크를 들고 있다가, 다음 청크를 그 남은 꼬리 위에 이어 붙이게 한다."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        #: 지금 처리 중인 관측의 timestep. 클라이언트가 실어 보낸 값이다.
        self.timestep: int | None = None
        #: 마지막으로 낸 청크. **후처리 전**의 값이어야 한다 — RTC의 유도는 정책이 내놓는
        #: 공간에서 이루어지고, 역정규화를 거친 값을 넣으면 엉뚱한 곳으로 끌어당긴다.
        self.chunk = None
        #: 그 청크의 첫 액션이 놓인 timestep.
        self.chunk_timestep: int | None = None
        #: 최근 추론이 걸린 프레임 수. 큐가 새 청크의 앞부분을 그만큼 버린다.
        self.delay = 0

    def left_over(self):
        """아직 실행되지 않은 앞 계획의 꼬리와, 그 사이 소비된 프레임 수.

        꼬리가 없으면 `(None, 0)`. 소비량을 함께 돌려주는 이유는 그것이 왕복 지연의
        **관측값**이기 때문이다 — `get_action_chunk`가 그것으로 고정 구간을 정한다.
        """
        if self.chunk is None or self.chunk_timestep is None or self.timestep is None:
            return None, 0
        consumed = self.timestep - self.chunk_timestep
        if consumed <= 0 or consumed >= self.chunk.shape[1]:
            return None, 0
        return self.chunk[:, consumed:, :], consumed

    def get_action_chunk(self, server, observation):
        guided = _rtc_mode(server.policy) != "none"
        previous, consumed = self.left_over() if guided else (None, 0)
        started = time.perf_counter()
        if previous is None:
            chunk = _original_get_action_chunk(server, observation)
        else:
            # 유도 구간은 꼬리보다 길 수 없고(lerobot이 스스로 줄인다), 지연은 그 안에
            # 들어와야 실제로 실행되는 부분에 제약이 남는다.
            horizon = min(RTC_EXECUTION_HORIZON, int(previous.shape[1]))
            # 얼려야 하는 프레임 수는 **추론이 걸린 시간이 아니라 왕복 전체**다.
            # `self.delay`는 추론 시간만 세므로 2026-09-09 실측에서 5프레임이었는데,
            # 클라이언트가 청크를 기다리는 동안 실제로 실행해 버린 액션은 중앙값 7,
            # 90퍼센타일 13프레임이었다(왕복 261ms = 7.8프레임: 추론 163ms에
            # 직렬화·전송·역직렬화·큐가 더해진다). 그 차이만큼 앞이 고정되지 않은 채
            # 실행돼서, 청크가 갈릴 때마다 명령이 튀었다 — 7틱(238ms)마다 4° 넘는
            # 계단이 났고 그 54%가 팔의 진행 방향과 반대였다.
            #
            # `consumed`는 그 왕복의 관측값이다. 추론 시간을 바닥으로 두고 둘 중 큰
            # 값을 쓴다. 넘겨 잡는 쪽이 안전하다 — 모자라면 튀고, 넘치면 앞 계획을
            # 조금 더 오래 따를 뿐이다.
            delay = max(0, min(max(self.delay, consumed), horizon - 1))
            chunk = server.policy.predict_action_chunk(
                observation, inference_delay=delay, prev_chunk_left_over=previous
            )
            if chunk.ndim != 3:
                chunk = chunk.unsqueeze(0)
            chunk = chunk[:, : server.actions_per_chunk, :]
        self._remember(server, chunk, time.perf_counter() - started)
        # 이어 붙였는지 아닌지는 로그로 남는다. 이 줄이 없으면 RTC가 켜졌다는 것과 실제로
        # 걸렸다는 것을 구별할 방법이 없다 — 꼬리가 없으면 켜져 있어도 걸리지 않는다.
        if previous is None:
            print("[soarm] 이어 붙일 앞 계획이 없어 새로 만듭니다", flush=True)
        else:
            print(
                f"[soarm] 앞 계획 {int(previous.shape[1])}프레임 위에 이어 붙였습니다 "
                f"(버려질 앞부분 {self.delay}프레임)",
                flush=True,
            )
        return chunk

    def _remember(self, server, chunk, seconds: float) -> None:
        fps = getattr(server.config, "fps", 30) or 30
        self.delay = max(0, math.ceil(seconds * fps))
        self.chunk = chunk.detach().clone()
        self.chunk_timestep = self.timestep


_RTC = _RealTimeChunking()


def _cached_get_policy_class(name: str) -> _CachedPolicyClass:
    return _CachedPolicyClass(_original_get_policy_class(name))


def _patched_predict_action_chunk(self, observation_t):
    # 이 관측이 놓인 자리를 기억해 둔다. 아래 `_get_action_chunk`가 앞 계획과 견주는 기준이다.
    _RTC.timestep = observation_t.get_timestep()
    return _original_predict_action_chunk(self, observation_t)


def _patched_get_action_chunk(self, observation):
    return _RTC.get_action_chunk(self, observation)


def _patched_reset_server(self):
    # 새 클라이언트가 붙었다. 앞 계획은 지난 시행의 것이므로 이어 붙일 대상이 아니다.
    _RTC.reset()
    return _original_reset_server(self)


def main() -> None:
    policy_server.get_policy_class = _cached_get_policy_class
    policy_server.PolicyServer._get_action_chunk = _patched_get_action_chunk
    policy_server.PolicyServer._predict_action_chunk = _patched_predict_action_chunk
    policy_server.PolicyServer._reset_server = _patched_reset_server
    # `serve`는 draccus로 감싸여 있어 `--host/--port/--fps`를 sys.argv에서 읽는다.
    policy_server.serve()


if __name__ == "__main__":
    main()
